from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.schemas.evaluation import EvaluationRequest, EvaluationResult
from app.services import evaluation_service

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationResult)
def run_evaluation(
    payload: EvaluationRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    result = evaluation_service.run_evaluation(
        db,
        owner_id=current_user.id,
        question=payload.question,
        document_ids=payload.documentIds,
        expected_keywords=payload.expectedKeywords,
        expected_sources=payload.expectedSources,
    )
    return EvaluationResult(**result)


@router.get("/results", response_model=List[EvaluationResult])
def get_results(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    runs = evaluation_service.list_evaluation_runs(db, owner_id=current_user.id)
    return [EvaluationResult(**run) for run in runs]
