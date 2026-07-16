from pathlib import Path

import pytest

from vibeflow.db import TaskRepository
from vibeflow.services import TaskService


@pytest.fixture
def service(tmp_path: Path) -> TaskService:
    repository = TaskRepository(tmp_path / "test.db")
    return TaskService(repository)


def test_create_task(service: TaskService) -> None:
    task = service.create_task("复习线性代数", "线性代数", "2026-07-20")

    assert task.id is not None
    assert task.title == "复习线性代数"
    assert task.status == "pending"


def test_title_cannot_be_empty(service: TaskService) -> None:
    with pytest.raises(ValueError, match="任务标题不能为空"):
        service.create_task("   ")


def test_due_date_format_validation(service: TaskService) -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        service.create_task("完成作业", due_date="2026/07/20")


def test_toggle_task_status(service: TaskService) -> None:
    task = service.create_task("完成统计学习题")

    completed = service.toggle_status(task.id)
    pending = service.toggle_status(task.id)

    assert completed.status == "completed"
    assert pending.status == "pending"


def test_statistics(service: TaskService) -> None:
    first = service.create_task("任务一")
    service.create_task("任务二")
    service.toggle_status(first.id)

    statistics = service.get_statistics()

    assert statistics == {
        "total": 2,
        "pending": 1,
        "completed": 1,
    }


def test_delete_task(service: TaskService) -> None:
    task = service.create_task("待删除任务")
    service.delete_task(task.id)

    assert service.list_tasks() == []
