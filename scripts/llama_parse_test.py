import nest_asyncio

nest_asyncio.apply()

from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="llx-MSElHmAZU1AVJ2OSYs0waFF3uaDnqbhUYYTGekQ7tCikDvbB",  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="text",  # "markdown" and "text" are available
    num_workers=4,  # if multiple files passed, split in `num_workers` API calls
    verbose=True,
    language="en",  # Optionally you can define a language, default=en
)

# sync
documents = parser.load_data("./Lebanese Situation Dictionary.pdf")

# Save the parsed results
with open('./Lebanese Situation Dictionary.md', 'w', encoding='utf-8') as f:
    for doc in documents:
        f.write(doc.text + '\n')

# sync batch
# documents = parser.load_data(["./my_file1.pdf", "./my_file2.pdf"])
