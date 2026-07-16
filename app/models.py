from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class Task:
    id: Optional[int]
    title: str
    course: str
    due_date: str
    status: str = "pending"
    created_at: str = ""
