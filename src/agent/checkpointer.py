from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from langgraph.checkpoint.mongodb import MongoDBSaver
import os

mongo_connection_string = os.environ["MONGODB_URI"]


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langgraph.checkpoint.base import BaseCheckpointSaver

# Marker key used inside the checkpoint document to flag channel values that
# have been serialized with ``serde.dumps_typed`` (typed blobs).  The base
# ``_deserialize_channel_values`` will recurse into the dict but leave the
# marker untouched because it doesn't match any known revival patterns.
_TYPED_MARKER = "__serde_typed__"


@asynccontextmanager
async def generate_checkpointer() -> AsyncIterator[BaseCheckpointSaver]:
    async with MongoDBSaver.from_conn_string(mongo_connection_string) as checkpointer:
        yield checkpointer
