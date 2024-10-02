# Contextual Retrieval

[Contextual Retrieval](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/contextual_retrieval.ipynb)

```python
prompt_document = """<document>
{WHOLE_DOCUMENT}
</document>"""

prompt_chunk = """Here is the chunk we want to situate within the whole document
<chunk>
{CHUNK_CONTENT}
</chunk>
Please give a short succinct context to situate this chunk within the overall document \
for the purposes of improving search retrieval of the chunk. \
Answer only with the succinct context and nothing else."""
```
