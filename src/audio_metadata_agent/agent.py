from langgraph.graph import END, StateGraph

from audio_metadata_agent.nodes.embedding_nodes import (
    generate_embedding_node,
    save_to_chroma_node,
    vector_db_lookup_node,
)
from audio_metadata_agent.nodes.metadata_nodes import (
    propose_corrections_node,
    read_metadata_node,
    write_metadata_node,
)

from .schemas.agent_state import AgentState

workflow = StateGraph(AgentState)

workflow.add_node("read_metadata", read_metadata_node)
workflow.add_node("generate_embedding", generate_embedding_node)
workflow.add_node("vector_db_lookup", vector_db_lookup_node)
workflow.add_node("propose_corrections", propose_corrections_node)
workflow.add_node("write_metadata", write_metadata_node)
workflow.add_node("save_to_chroma", save_to_chroma_node)

workflow.set_entry_point("read_metadata")
workflow.add_edge("read_metadata", "generate_embedding")
workflow.add_edge("generate_embedding", "vector_db_lookup")
workflow.add_edge("vector_db_lookup", "propose_corrections")
workflow.add_edge("propose_corrections", "write_metadata")
workflow.add_edge("write_metadata", "save_to_chroma")
workflow.add_edge("save_to_chroma", END)

graph = workflow.compile()
