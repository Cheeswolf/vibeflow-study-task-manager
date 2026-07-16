from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from app.models import Task


class TaskRepository:
    def __init__(self, db_path: str | Path = "vibeflow.db") -> None:
        self.db_path = str(db_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    course TEXT NOT NULL DEFAULT '',
                    due_date TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                '''
            )

    def add(self, task: Task) -> Task:
        with self._connect() as connection:
            cursor = connection.execute(
                '''
                INSERT INTO tasks (title, course, due_date, status)
                VALUES (?, ?, ?, ?)
                ''',
                (task.title, task.course, task.due_date, task.status),
            )
            task_id = int(cursor.lastrowid)
        return self.get(task_id)

    def get(self, task_id: int) -> Task:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"任务不存在：{task_id}")
        return self._row_to_task(row)

    def list_all(self, status: str | None = None) -> list[Task]:
        query = "SELECT * FROM tasks"
        params: Iterable[str] = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY id DESC"

        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [self._row_to_task(row) for row in rows]

    def update_status(self, task_id: int, status: str) -> Task:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE tasks SET status = ? WHERE id = ?",
                (status, task_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"任务不存在：{task_id}")
        return self.get(task_id)

    def delete(self, task_id: int) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM tasks WHERE id = ?",
                (task_id,),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"任务不存在：{task_id}")

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"],
            title=row["title"],
            course=row["course"],
            due_date=row["due_date"],
            status=row["status"],
            created_at=row["created_at"],
        )
