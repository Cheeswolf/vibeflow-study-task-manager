from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from app.db import TaskRepository
from app.services import TaskService


class VibeFlowApp:
    STATUS_LABELS = {
        "all": "全部",
        "pending": "待完成",
        "completed": "已完成",
    }

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("VibeFlow 学习任务管理器")
        self.root.geometry("900x560")
        self.root.minsize(760, 480)

        self.service = TaskService(TaskRepository())
        self.filter_value = tk.StringVar(value="all")
        self.title_value = tk.StringVar()
        self.course_value = tk.StringVar()
        self.due_date_value = tk.StringVar()
        self.stats_value = tk.StringVar()

        self._build_ui()
        self.refresh_tasks()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            container,
            text="VibeFlow 学习任务管理器",
            font=("", 18, "bold"),
        )
        title.pack(anchor=tk.W, pady=(0, 12))

        form = ttk.LabelFrame(container, text="新建任务", padding=12)
        form.pack(fill=tk.X)

        ttk.Label(form, text="任务标题").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(form, textvariable=self.title_value, width=32).grid(
            row=1, column=0, padx=(0, 10), sticky=tk.EW
        )

        ttk.Label(form, text="所属课程").grid(row=0, column=1, sticky=tk.W)
        ttk.Entry(form, textvariable=self.course_value, width=22).grid(
            row=1, column=1, padx=(0, 10), sticky=tk.EW
        )

        ttk.Label(form, text="截止日期").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(form, textvariable=self.due_date_value, width=16).grid(
            row=1, column=2, padx=(0, 10), sticky=tk.EW
        )

        ttk.Button(form, text="添加任务", command=self.add_task).grid(
            row=1, column=3, sticky=tk.EW
        )

        form.columnconfigure(0, weight=2)
        form.columnconfigure(1, weight=1)
        form.columnconfigure(2, weight=1)

        toolbar = ttk.Frame(container)
        toolbar.pack(fill=tk.X, pady=12)

        ttk.Label(toolbar, text="筛选：").pack(side=tk.LEFT)
        for value, label in self.STATUS_LABELS.items():
            ttk.Radiobutton(
                toolbar,
                text=label,
                variable=self.filter_value,
                value=value,
                command=self.refresh_tasks,
            ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(toolbar, textvariable=self.stats_value).pack(side=tk.RIGHT)

        table_frame = ttk.Frame(container)
        table_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("id", "title", "course", "due_date", "status")
        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
        )

        headers = {
            "id": "ID",
            "title": "任务",
            "course": "课程",
            "due_date": "截止日期",
            "status": "状态",
        }
        widths = {
            "id": 60,
            "title": 330,
            "course": 160,
            "due_date": 120,
            "status": 100,
        }

        for column in columns:
            self.tree.heading(column, text=headers[column])
            self.tree.column(column, width=widths[column], anchor=tk.CENTER)
        self.tree.column("title", anchor=tk.W)

        scrollbar = ttk.Scrollbar(
            table_frame,
            orient=tk.VERTICAL,
            command=self.tree.yview,
        )
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        actions = ttk.Frame(container)
        actions.pack(fill=tk.X, pady=(12, 0))

        ttk.Button(
            actions,
            text="切换完成状态",
            command=self.toggle_selected_task,
        ).pack(side=tk.LEFT)

        ttk.Button(
            actions,
            text="删除任务",
            command=self.delete_selected_task,
        ).pack(side=tk.LEFT, padx=8)

        ttk.Button(
            actions,
            text="刷新",
            command=self.refresh_tasks,
        ).pack(side=tk.RIGHT)

    def add_task(self) -> None:
        try:
            self.service.create_task(
                self.title_value.get(),
                self.course_value.get(),
                self.due_date_value.get(),
            )
        except ValueError as exc:
            messagebox.showerror("无法添加任务", str(exc))
            return

        self.title_value.set("")
        self.course_value.set("")
        self.due_date_value.set("")
        self.refresh_tasks()

    def refresh_tasks(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        selected_filter = self.filter_value.get()
        status = None if selected_filter == "all" else selected_filter
        tasks = self.service.list_tasks(status)

        for task in tasks:
            status_label = "已完成" if task.status == "completed" else "待完成"
            self.tree.insert(
                "",
                tk.END,
                values=(
                    task.id,
                    task.title,
                    task.course,
                    task.due_date,
                    status_label,
                ),
            )

        stats = self.service.get_statistics()
        self.stats_value.set(
            f"全部 {stats['total']}｜待完成 {stats['pending']}｜已完成 {stats['completed']}"
        )

    def _get_selected_task_id(self) -> int | None:
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("未选择任务", "请先选择一条任务")
            return None
        values = self.tree.item(selection[0], "values")
        return int(values[0])

    def toggle_selected_task(self) -> None:
        task_id = self._get_selected_task_id()
        if task_id is None:
            return
        try:
            self.service.toggle_status(task_id)
        except ValueError as exc:
            messagebox.showerror("操作失败", str(exc))
            return
        self.refresh_tasks()

    def delete_selected_task(self) -> None:
        task_id = self._get_selected_task_id()
        if task_id is None:
            return

        confirmed = messagebox.askyesno("确认删除", "确定删除选中的任务吗？")
        if not confirmed:
            return

        try:
            self.service.delete_task(task_id)
        except ValueError as exc:
            messagebox.showerror("删除失败", str(exc))
            return
        self.refresh_tasks()


def main() -> None:
    root = tk.Tk()
    app = VibeFlowApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
