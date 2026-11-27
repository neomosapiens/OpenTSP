"""Load the maintained TinyStories numerical reference for serving tests."""
def load():
    from opentsp_vllm.models.tinystories import paged_reference, weights, config
    return paged_reference, weights, config
