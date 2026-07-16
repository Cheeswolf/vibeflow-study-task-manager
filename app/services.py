from __future__ import annotations

from datetime import datetime

from app.db import TaskRepository
from app.models import Task


class TaskService:
    VALID_STATUSES = {"pending", "completed"}

    def __init__(self, repository: TaskRepository) -> None:
        self.repository = repository

    def create_task(self, title: str, course: str = "", due_date: str = "") -> Task:
        title = title.strip()
        course = course.strip()
        due_date = due_date.strip()

        if not title:
            raise ValueError("任务标题不能为空")

        if due_date:
            try:
                datetime.strptime(due_date, "%Y-%m-%d")
            except ValueError as exc:
                raise ValueError("截止日期必须使用 YYYY-MM-DD 格式") from exc

        return self.repository.add(
            Task(
                id=None,
                title=title,
                course=course,
                due_date=due_date,
                status="pending",
            )
        )

    def list_tasks(self, status: str | None = None) -> list[Task]:
        if status is not None and status not in self.VALID_STATUSES:
            raise ValueError(f"无效状态：{status}")
        return self.repository.list_all(status)

    def toggle_status(self, task_id: int) -> Task:
        current = self.repository.get(task_id)
        new_status = "completed" if current.status == "pending" else "pending"
        return self.repository.update_status(task_id, new_status)

    def delete_task(self, task_id: int) -> None:
        self.repository.delete(task_id)

    def get_statistics(self) -> dict[str, int]:
        tasks = self.repository.list_all()
        completed = sum(task.status == "completed" for task in tasks)
        pending = len(tasks) - completed
        return {
            "total": len(tasks),
            "pending": pending,
            "completed": completed,
        }
