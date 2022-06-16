import uvicorn
from init import dataset, category_labels, source_labels, my_models, process_pipeline, trim_middle_string, trim_middle_array, model_names, model_accuracies, arr_to_s, timestamp
from typing import List
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import io
app = FastAPI()
app.mount("/public", StaticFiles(directory="public"))
app.add_middleware(CORSMiddleware, allow_origins = ["*"], allow_methods = ["*"],
        allow_headers = ["*"], allow_credentials = True, expose_headers = ["Content-Disposition"])

class Model(BaseModel):
    key: str
    name: str
    accuracy: float

class Description(BaseModel):
    length: int
    models: List[Model]

class Row(BaseModel):
    text: str
    category: str
    source: str
    url: str

class Entry(BaseModel):
    text: str
    model_name: str
    export: bool = False

class Result(BaseModel):
    preprocess: str
    tokenization: str
    vectorization: str
    prediction: str

def iterfile(file_path):
    with open(file_path, mode="rb") as file_like:
        yield from file_like

@app.get("/", response_class=HTMLResponse)
def get_homepage():
    with open('./index.html', mode="rb") as file:
        return file.read()

@app.get("/dataset", response_model = Description)
def read_dataset():
    return Description(
            length = len(dataset),
            models = [Model(key = key, name = model_names[key], accuracy = model_accuracies[key]) for key in model_names.keys()]
            )

@app.get("/dataset/{index}", response_model = Row)
def read_dataset_record(index: int):
    row = dataset.loc[index]
    return Row(
        text = row.text,
        category = category_labels[row.category_id],
        source = source_labels[row.source_id],
        url = row.url
        )

@app.post("/predict/", response_model = Result)
def create_item(entry: Entry):
    process_pipeline.process(entry.text, my_models[entry.model_name])
    if entry.export:
        df = pd.DataFrame({
            'input': [entry.text], 'model': [entry.model_name],
            'preprocess': [process_pipeline.preprocess],
            'tokenization': [process_pipeline.tokenization],
            'vectorization': [arr_to_s(process_pipeline.feature_selection.toarray()[0])],
            'prediction': [category_labels[int(process_pipeline.prediction[0])]]
            })
        response = StreamingResponse(io.StringIO(df.to_csv(index=False)), media_type="text/csv")
        response.headers['Content-Disposition'] = 'attachment; filename=prediction_%s.csv' % timestamp()
        return response
    else:
        return Result(
                preprocess = trim_middle_string(process_pipeline.preprocess, 200),
                tokenization = trim_middle_array(process_pipeline.tokenization),
                vectorization = trim_middle_array(process_pipeline.feature_selection.toarray()[0]),
                prediction = category_labels[int(process_pipeline.prediction[0])]
                )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
