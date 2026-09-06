"""Durable task state: it is written, it is owned, and it survives a restart."""

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents import store
from app.agents.state import Evaluation, StepStatus, SubTask, TaskState, TaskStatus


@pytest.fixture
def db(monkeypatch):
    """An isolated DB for persistence tests.

    store.save opens its own session through SessionLocal rather than taking
    one, precisely so a write can never be rolled back by a caller's
    transaction — so the module-level factory is what has to be redirected.
    """
    from app import models  # noqa: F401  — registers AgentTask on Base before create_all
    from app.db import Base

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(store, "SessionLocal", Session)
    return Session()


def _state(**kw):
    st = TaskState(user_goal="Research competitors and compare pricing", chat_id="c1", **kw)
    st.plan = [
        SubTask(id=1, agent="research", task="find competitors", tool="web_search"),
        SubTask(id=2, agent="analyse", task="compare", depends_on=[1]),
    ]
    return st


# ── Writing ──────────────────────────────────────────────────────────────────

def test_a_task_is_written_with_every_field_that_matters(db):
    st = _state(user_id=7)
    st.plan[0].status = StepStatus.DONE
    st.agent_outputs = {1: "competitor list"}
    st.tool_outputs = {"web_search": "raw results"}
    st.failures = ["step 2 timed out"]
    st.retry_count = 1
    st.evaluation = Evaluation(complete=True, grounded=True, verdict="pass")
    st.final_result = "the report"
    st.status = TaskStatus.COMPLETED

    store.save(st, st.user_id)
    row = store.get(db, 7, st.task_id)

    assert row["task_id"] == st.task_id
    assert row["goal"].startswith("Research competitors")
    assert row["status"] == "COMPLETED"
    assert row["retry_count"] == 1
    assert row["final_result"] == "the report"
    assert row["agent_outputs"]["1"] == "competitor list"
    assert row["tool_outputs"]["web_search"] == "raw results"
    assert row["failures"] == ["step 2 timed out"]
    assert row["evaluation"]["verdict"] == "pass"
    assert [s["agent"] for s in row["plan"]] == ["research", "analyse"]
    assert row["created_at"] and row["updated_at"]


def test_saving_twice_updates_the_same_row(db):
    from app.models import AgentTask

    st = _state(user_id=7)
    store.save(st, 7)
    st.status = TaskStatus.COMPLETED
    st.final_result = "done"
    store.save(st, 7)

    assert db.query(AgentTask).count() == 1
    assert store.get(db, 7, st.task_id)["status"] == "COMPLETED"


def test_an_unowned_task_is_not_written(db):
    """No owner means no way to read it back safely, so there is no value in
    the row and some cost in keeping it."""
    from app.models import AgentTask

    store.save(_state(), None)
    assert db.query(AgentTask).count() == 0


def test_a_write_failure_never_propagates(db, monkeypatch):
    """The answer is the product; the record is bookkeeping. A task that
    produced a good answer must not fail because a write failed."""
    class _Broken:
        def __call__(self):
            raise RuntimeError("database is down")

    monkeypatch.setattr(store, "SessionLocal", _Broken())
    store.save(_state(user_id=7), 7)   # must not raise


# ── Secrecy ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("secret", [
    "sk-abcdefghijklmnopqrstuvwx",
    "nvapi-abcdefghijklmnopqrstuvwx",
    "gsk_abcdefghijklmnopqrstuvwx",
])
def test_credentials_are_redacted_on_the_way_to_the_database(db, secret):
    """A value can enter the state through paths that never went through
    write_working — a step's task text, a failure string, the user's own goal —
    so redaction runs again at the boundary."""
    st = TaskState(user_goal=f"look at {secret} please", chat_id="c1", user_id=7)
    st.plan = [SubTask(id=1, agent="chat", task="x")]
    st.agent_outputs = {1: f"found {secret} in the file"}
    st.failures = [f"auth failed with {secret}"]
    st.final_result = f"the key is {secret}"

    store.save(st, 7)
    blob = json.dumps(store.get(db, 7, st.task_id))

    assert secret not in blob
    assert "[redacted]" in blob


# ── Ownership ────────────────────────────────────────────────────────────────

def test_another_users_task_is_invisible(db):
    """Filtered on user_id as well as id, so knowing or guessing an id is not
    enough to read someone else's run."""
    st = _state(user_id=7)
    store.save(st, 7)

    assert store.get(db, 7, st.task_id) is not None
    assert store.get(db, 99, st.task_id) is None


def test_the_listing_is_scoped_to_the_caller(db):
    mine = _state(user_id=7)
    theirs = _state(user_id=99)
    store.save(mine, 7)
    store.save(theirs, 99)

    ids = [t["task_id"] for t in store.list_for_user(db, 7)]
    assert ids == [mine.task_id]


def test_the_listing_omits_bulky_step_output(db):
    """A routine listing must not carry every step's full text."""
    st = _state(user_id=7)
    st.agent_outputs = {1: "x" * 5000}
    store.save(st, 7)

    row = store.list_for_user(db, 7)[0]
    assert "agent_outputs" not in row and "tool_outputs" not in row
    assert row["status"] and row["plan"]


def test_the_listing_is_bounded(db):
    for _ in range(5):
        store.save(_state(user_id=7), 7)
    assert len(store.list_for_user(db, 7, limit=2)) == 2
    assert len(store.list_for_user(db, 7, limit=10_000)) <= 100


# ── Restart recovery ─────────────────────────────────────────────────────────

def test_tasks_running_at_shutdown_are_closed_out_not_left_hanging(db):
    """The client's stream died with the process, so nothing is resumable. A
    row left RUNNING would show a task that never finishes and never fails."""
    running = _state(user_id=7)
    running.status = TaskStatus.RUNNING
    running.agent_outputs = {1: "partial work worth keeping"}
    finished = _state(user_id=7)
    finished.status = TaskStatus.COMPLETED
    store.save(running, 7)
    store.save(finished, 7)

    assert store.mark_interrupted_tasks() == 1

    after = store.get(db, 7, running.task_id)
    assert after["status"] == "FAILED"
    assert "Interrupted by a server restart." in after["failures"]
    assert after["agent_outputs"]["1"] == "partial work worth keeping"
    assert store.get(db, 7, finished.task_id)["status"] == "COMPLETED"


@pytest.mark.parametrize("status", list(store.NON_TERMINAL))
def test_every_non_terminal_status_is_closed_out(db, status):
    st = _state(user_id=7)
    st.status = status
    store.save(st, 7)
    store.mark_interrupted_tasks()
    assert store.get(db, 7, st.task_id)["status"] == "FAILED"


def test_recovery_is_idempotent(db):
    st = _state(user_id=7)
    st.status = TaskStatus.RUNNING
    store.save(st, 7)

    assert store.mark_interrupted_tasks() == 1
    assert store.mark_interrupted_tasks() == 0


def test_corrupt_json_in_a_row_does_not_break_reads(db):
    """Rows outlive code. A column that cannot be parsed must degrade to a
    default, not take down the listing."""
    from app.models import AgentTask

    st = _state(user_id=7)
    store.save(st, 7)
    row = db.get(AgentTask, st.task_id)
    row.plan = "{not json"
    row.agent_outputs = ""
    db.commit()

    got = store.get(db, 7, st.task_id)
    assert got["plan"] == [] and got["agent_outputs"] == {}
