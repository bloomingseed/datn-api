from init import dataset, category_labels, source_labels,vectorizer, selector, my_models
from typing import Union
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
    
    
@app.get("/dataset")
def read_dataset():
    return {'length': len(dataset)}
    
@app.get("/dataset/{index}")
def read_dataset_record(index: int):
    row = dataset.loc[index]
    return {
        'text': row.text,
        'category': category_labels[row.category_id],
        'source': source_labels[row.source_id]
    }
    
