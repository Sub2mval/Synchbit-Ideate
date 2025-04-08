import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.utilities import SQLDatabase

DATABASE_URL = os.environ["DATABASE_URL"]
db = SQLDatabase.from_uri(DATABASE_URL)
print(db.dialect)
#print(db.get_usable_table_names())