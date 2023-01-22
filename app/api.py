from typing import Optional
from fastapi import FastAPI
from fastapi import Form, Query
from fastapi.responses import JSONResponse
from app.utils import *
from app.model import AnimeContentRecommender
import uvicorn

recommender = AnimeContentRecommender()

app = FastAPI()

@app.post('/anime_recommendations')
async def recommend(
    anime: str = Form(...),
    sort: Optional[str] = Query('Similarity Score', enum=['Rating', 'Similarity Score']),
    n_recoms: Optional[int] = Form(10),
    threshold: Optional[float] = Form(0.3)
):
    if sort == 'Similarity Score':
        sort = 'Similarity_Scores'
    else:
        sort = 'Score'

    response = get_recommendations(anime, sort, n_recoms, threshold, recommender)
    if type(response) == list:
        return JSONResponse(status_code=200, content={'errors': response})
    else:
        return JSONResponse(status_code=200, content={'recommendations': response.reset_index(drop=True).to_dict('index')})

if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port=8080)
