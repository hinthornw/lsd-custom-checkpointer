"""Test fixture: Redis-backed custom checkpointer.

Extends AsyncRedisSaver with typed channel-value serialization (preserving
Python types such as int dict keys that Redis JSON would otherwise lose).

The adapter provides generic fallbacks for acopy_thread and aprune, so only
serialization overrides and adelete_for_runs are needed here.

adelete_for_runs is implemented here because the adapter has no safe generic
fallback — the only base primitive (adelete_thread) wipes the entire thread.
"""

from __future__ import annotations

import base64
import binascii
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import orjson
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.checkpoint.redis.base import safely_decode

from langgraph_api import config

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint

# Marker key used inside the checkpoint document to flag channel values that
# have been serialized with ``serde.dumps_typed`` (typed blobs).  The base
# ``_deserialize_channel_values`` will recurse into the dict but leave the
# marker untouched because it doesn't match any known revival patterns.
_TYPED_MARKER = "__serde_typed__"


@asynccontextmanager
async def generate_checkpointer() -> AsyncIterator[BaseCheckpointSaver]:
    if not config.REDIS_URI:
        raise ValueError("REDIS_URI must be set to use the Redis checkpointer.")

    class _RedisCheckpointer(AsyncRedisSaver):
        # -- serialization overrides ------------------------------------------

        def _recursive_deserialize(self, obj: Any) -> Any:
            """Intercept typed-blob markers before base-class processing.

            The async ``aget_tuple`` calls ``_recursive_deserialize`` (not
            ``_deserialize_channel_values``) on loaded channel values, so we
            must hook in here to decode our ``__serde_typed__`` envelopes.
            """
            if isinstance(obj, dict) and obj.get(_TYPED_MARKER):
                blob = base64.b64decode(obj["data"])
                return self.serde.loads_typed((obj["encoding"], blob))
            return super()._recursive_deserialize(obj)

        def _dump_checkpoint(self, checkpoint: Checkpoint) -> dict[str, Any]:
            """Serialize checkpoint with typed channel values.

            Each channel value is individually serialized via
            ``serde.dumps_typed`` so that non-JSON types (e.g. dicts with int
            keys) survive the Redis-JSON round-trip.  Everything else is
            delegated to the base class logic.
            """
            channel_values = checkpoint.get("channel_values", {})
            rest = {k: v for k, v in checkpoint.items() if k != "channel_values"}

            # Let the base serde serialise the skeleton (id, ts, versions …)
            type_, data = self.serde.dumps_typed(rest)
            if type_ == "json":
                checkpoint_data = cast("dict", orjson.loads(data))
            else:
                checkpoint_data = cast("dict", self.serde.loads_typed((type_, data)))

            # Serialize each channel value individually so types are preserved.
            typed_cv: dict[str, Any] = {}
            for channel, value in channel_values.items():
                enc, blob = self.serde.dumps_typed(value)
                typed_cv[channel] = {
                    _TYPED_MARKER: True,
                    "encoding": enc,
                    "data": base64.b64encode(blob).decode("ascii"),
                }
            checkpoint_data["channel_values"] = typed_cv

            if "channel_versions" in checkpoint_data:
                checkpoint_data["channel_versions"] = {
                    k: str(v) for k, v in checkpoint_data["channel_versions"].items()
                }

            return {"type": type_, **checkpoint_data, "pending_sends": []}

        def _deserialize_channel_values(
            self, channel_values: dict[str, Any]
        ) -> dict[str, Any]:
            """Deserialize typed channel values back to Python objects."""
            if not channel_values:
                return {}
            result: dict[str, Any] = {}
            for channel, value in channel_values.items():
                if isinstance(value, dict) and value.get(_TYPED_MARKER):
                    blob = base64.b64decode(value["data"])
                    result[channel] = self.serde.loads_typed((value["encoding"], blob))
                else:
                    # Legacy / inline format — delegate to base class
                    deserialized = super()._deserialize_channel_values({channel: value})
                    result[channel] = deserialized.get(channel, value)
            return result

        def _load_checkpoint(
            self,
            checkpoint: dict[str, Any] | str,
            channel_values: dict[str, Any],
            pending_sends: list[Any],
        ) -> Checkpoint:
            """Load checkpoint, decoding base64 pending-send blobs."""
            if not checkpoint:
                return {}

            loaded = (
                checkpoint
                if isinstance(checkpoint, dict)
                else cast("dict", orjson.loads(checkpoint))
            )

            decoded_sends = []
            for c, b in pending_sends or []:
                if isinstance(b, str):
                    try:
                        b = base64.b64decode(b.encode("utf-8"))
                    except (binascii.Error, ValueError):
                        b = b.encode("utf-8")
                decoded_sends.append(self.serde.loads_typed((safely_decode(c), b)))

            return {
                **loaded,
                "pending_sends": decoded_sends,
                "channel_values": channel_values,
            }

        # -- adelete_for_runs: kept as custom impl to avoid race condition ----
        # The generic fallback uses adelete_thread + re-insert which is not
        # safe when another run is concurrently writing to the same thread
        # (e.g. rollback of run 1 while run 2 is already executing).

        async def adelete_for_runs(self, run_ids: Iterable[str]) -> None:
            """Delete checkpoints belonging to the given run IDs."""
            run_id_set = {str(r) for r in run_ids}
            to_delete: list[tuple[str, str, str]] = []
            for run_id in run_id_set:
                async for item in self.alist(None, filter={"run_id": run_id}):
                    cfg = item.config["configurable"]
                    to_delete.append(
                        (
                            cfg["thread_id"],
                            cfg.get("checkpoint_ns", ""),
                            cfg.get("checkpoint_id", ""),
                        )
                    )
            for thread_id, checkpoint_ns, checkpoint_id in to_delete:
                key = self._make_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
                await self._redis.delete(key)

        def _make_checkpoint_key(
            self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
        ) -> str:
            """Build the Redis key for a checkpoint document."""
            from langgraph.checkpoint.redis.util import (  # noqa: PLC0415
                to_storage_safe_id,
                to_storage_safe_str,
            )

            safe_tid = to_storage_safe_id(thread_id)
            safe_ns = to_storage_safe_str(checkpoint_ns)
            safe_cid = to_storage_safe_id(checkpoint_id)
            sep = ":"
            return (
                f"{self._checkpoint_prefix}{sep}{safe_tid}{sep}{safe_ns}{sep}{safe_cid}"
            )

    checkpointer_cm = _RedisCheckpointer.from_conn_string(config.REDIS_URI)
    async with checkpointer_cm as checkpointer:
        await checkpointer.setup()
        yield checkpointer
