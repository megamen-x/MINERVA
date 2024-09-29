# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Indexing-Engine to Query Read Adapters.

The parts of these functions that do type adaptation, renaming, collating, etc. should eventually go away.
Ideally this is just a straight read-thorugh into the object model.
"""

from typing import cast

import pandas as pd

from graphrag.model import CommunityReport, Covariate, Entity, Relationship, TextUnit
from graphrag.query.input.loaders.dfs import (
    read_community_reports,
    read_entities,
)

def read_indexer_reports(
    final_community_reports: pd.DataFrame,
    final_nodes: pd.DataFrame,
    community_level: int,
) -> list[CommunityReport]:
    """Read in the Community Reports from the raw indexing outputs."""
    report_df = final_community_reports
    entity_df = final_nodes
    entity_df = _filter_under_community_level(entity_df, community_level)
    entity_df["community"] = entity_df["community"].fillna(-1)
    entity_df["community"] = entity_df["community"].astype(int)

    entity_df = entity_df.groupby(["title"]).agg({"community": "max"}).reset_index()
    entity_df["community"] = entity_df["community"].astype(str)
    filtered_community_df = entity_df["community"].drop_duplicates()

    report_df = _filter_under_community_level(report_df, community_level)
    report_df = report_df.merge(filtered_community_df, on="community", how="inner")

    return read_community_reports(
        df=report_df,
        id_col="community",
        short_id_col="community",
        summary_embedding_col=None,
        content_embedding_col=None,
    )


def read_indexer_entities(
    final_nodes: pd.DataFrame,
    final_entities: pd.DataFrame,
    community_level: int,
) -> list[Entity]:
    """Read in the Entities from the raw indexing outputs."""
    entity_df = final_nodes
    entity_embedding_df = final_entities

    entity_df = _filter_under_community_level(entity_df, community_level)
    entity_df = cast(pd.DataFrame, entity_df[["title", "degree", "community"]]).rename(
        columns={"title": "name", "degree": "rank"}
    )

    entity_df["community"] = entity_df["community"].fillna(-1)
    entity_df["community"] = entity_df["community"].astype(int)
    entity_df["rank"] = entity_df["rank"].astype(int)

    # for duplicate entities, keep the one with the highest community level
    entity_df = (
        entity_df.groupby(["name", "rank"]).agg({"community": "max"}).reset_index()
    )
    entity_df["community"] = entity_df["community"].apply(lambda x: [str(x)])
    entity_df = entity_df.merge(
        entity_embedding_df, on="name", how="inner"
    ).drop_duplicates(subset=["name"])

    # read entity dataframe to knowledge model objects
    return read_entities(
        df=entity_df,
        id_col="id",
        title_col="name",
        type_col="type",
        short_id_col="human_readable_id",
        description_col="description",
        community_col="community",
        rank_col="rank",
        name_embedding_col=None,
        description_embedding_col="description_embedding",
        graph_embedding_col=None,
        text_unit_ids_col="text_unit_ids",
        document_ids_col=None,
    )


def _filter_under_community_level(
    df: pd.DataFrame, community_level: int
) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        df[df.level <= community_level],
    )
