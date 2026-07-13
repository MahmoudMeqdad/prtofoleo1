from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class QueryDecision:
    original_query: str
    rewritten_query: str
    intent: str
    route: str
    clarification_question: str | None = None
    structured_answer: str | None = None
    course_code: str | None = None


class ResearchPlatform:
    """
    الطبقة البحثية لمساعد المساقات الجامعية.

    توفر:
    - قاعدة SQLite للبيانات الأكاديمية المنظمة.
    - تتبع حالة الحوار والكيانات.
    - إعادة صياغة الأسئلة التابعة للسياق.
    - اكتشاف الغموض وطرح سؤال توضيحي.
    - توجيه الأسئلة إلى SQL أو RAG.
    - تسجيل مؤشرات التجربة والتقييم.
    """

    COURSE_CODE_PATTERN = re.compile(
        r"\b([A-Za-z]{2,6})[\s\-]?(\d{2,4})\b",
        flags=re.IGNORECASE,
    )

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self._initialize_database()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS instructors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT,
                    office TEXT,
                    office_hours TEXT
                );

                CREATE TABLE IF NOT EXISTS courses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    name_ar TEXT NOT NULL,
                    name_en TEXT,
                    description TEXT,
                    credit_hours INTEGER,
                    instructor_id INTEGER,
                    FOREIGN KEY (instructor_id) REFERENCES instructors(id)
                );

                CREATE TABLE IF NOT EXISTS schedules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    course_id INTEGER NOT NULL,
                    day TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    room TEXT,
                    semester TEXT,
                    FOREIGN KEY (course_id) REFERENCES courses(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS prerequisites (
                    course_id INTEGER NOT NULL,
                    prerequisite_course_id INTEGER NOT NULL,
                    PRIMARY KEY (course_id, prerequisite_course_id),
                    FOREIGN KEY (course_id) REFERENCES courses(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (prerequisite_course_id) REFERENCES courses(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS conversation_state (
                    session_id TEXT PRIMARY KEY,
                    last_course_code TEXT,
                    last_intent TEXT,
                    pending_clarification TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS research_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    original_query TEXT NOT NULL,
                    rewritten_query TEXT,
                    intent TEXT,
                    route TEXT,
                    course_code TEXT,
                    clarification_used INTEGER DEFAULT 0,
                    retrieved_count INTEGER DEFAULT 0,
                    response_time_ms REAL,
                    answer_preview TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_id INTEGER,
                    session_id TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                    comment TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (log_id) REFERENCES research_logs(id)
                );

                CREATE TABLE IF NOT EXISTS deep_retrieval_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    research_log_id INTEGER,
                    session_id TEXT NOT NULL,
                    query_variants TEXT,
                    stages INTEGER DEFAULT 1,
                    candidate_count INTEGER DEFAULT 0,
                    evidence_count INTEGER DEFAULT 0,
                    sufficient INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0,
                    missing_evidence TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (research_log_id) REFERENCES research_logs(id)
                );
                """
            )

    # -----------------------------------------------------
    # إدارة بيانات المساقات
    # -----------------------------------------------------

    def seed_demo_data(self) -> int:
        """يضيف بيانات تجريبية قابلة للحذف والاستبدال ببيانات الجامعة."""
        with self.connect() as db:
            existing = db.execute("SELECT COUNT(*) FROM courses").fetchone()[0]
            if existing:
                return 0

            instructors = [
                ("د. أحمد منصور", "ahmad@example.edu", "B-203", "الأحد 11:00–13:00"),
                ("د. سارة خليل", "sara@example.edu", "C-115", "الثلاثاء 10:00–12:00"),
                ("د. محمود سالم", "mahmoud@example.edu", "A-310", "الأربعاء 12:00–14:00"),
            ]
            db.executemany(
                "INSERT INTO instructors(name,email,office,office_hours) VALUES(?,?,?,?)",
                instructors,
            )

            courses = [
                ("CS101", "مقدمة في علوم الحاسوب", "Introduction to Computer Science",
                 "مفاهيم البرمجة والخوارزميات الأساسية.", 3, 1),
                ("CS205", "قواعد البيانات", "Database Systems",
                 "النمذجة العلائقية وSQL وتصميم قواعد البيانات.", 3, 2),
                ("CS310", "الذكاء الاصطناعي", "Artificial Intelligence",
                 "البحث والاستدلال والتعلم الآلي ومبادئ الأنظمة الذكية.", 3, 3),
            ]
            db.executemany(
                """
                INSERT INTO courses(
                    code,name_ar,name_en,description,credit_hours,instructor_id
                ) VALUES(?,?,?,?,?,?)
                """,
                courses,
            )

            rows = {
                row["code"]: row["id"]
                for row in db.execute("SELECT id,code FROM courses")
            }

            schedules = [
                (rows["CS101"], "الأحد", "09:00", "10:30", "A-101", "2026-1"),
                (rows["CS101"], "الثلاثاء", "09:00", "10:30", "A-101", "2026-1"),
                (rows["CS205"], "الاثنين", "11:00", "12:30", "B-204", "2026-1"),
                (rows["CS205"], "الأربعاء", "11:00", "12:30", "B-204", "2026-1"),
                (rows["CS310"], "الأحد", "13:00", "14:30", "C-301", "2026-1"),
                (rows["CS310"], "الخميس", "13:00", "14:30", "C-301", "2026-1"),
            ]
            db.executemany(
                """
                INSERT INTO schedules(
                    course_id,day,start_time,end_time,room,semester
                ) VALUES(?,?,?,?,?,?)
                """,
                schedules,
            )
            db.execute(
                """
                INSERT INTO prerequisites(course_id, prerequisite_course_id)
                VALUES(?,?)
                """,
                (rows["CS310"], rows["CS101"]),
            )
            return len(courses)

    def list_courses(self) -> list[sqlite3.Row]:
        with self.connect() as db:
            return db.execute(
                """
                SELECT c.code,c.name_ar,c.name_en,i.name AS instructor_name
                FROM courses c
                LEFT JOIN instructors i ON i.id=c.instructor_id
                ORDER BY c.code
                """
            ).fetchall()

    def find_course(self, reference: str) -> sqlite3.Row | None:
        reference = reference.strip()
        if not reference:
            return None

        normalized = reference.replace("-", "").replace(" ", "").upper()
        with self.connect() as db:
            exact = db.execute(
                """
                SELECT c.*, i.name AS instructor_name, i.email,
                       i.office, i.office_hours
                FROM courses c
                LEFT JOIN instructors i ON i.id=c.instructor_id
                WHERE REPLACE(REPLACE(UPPER(c.code),'-',''),' ','')=?
                """,
                (normalized,),
            ).fetchone()
            if exact:
                return exact

            matches = db.execute(
                """
                SELECT c.*, i.name AS instructor_name, i.email,
                       i.office, i.office_hours
                FROM courses c
                LEFT JOIN instructors i ON i.id=c.instructor_id
                WHERE c.name_ar LIKE ? OR c.name_en LIKE ?
                ORDER BY LENGTH(c.name_ar)
                """,
                (f"%{reference}%", f"%{reference}%"),
            ).fetchall()
            return matches[0] if len(matches) == 1 else None

    def course_details(self, code: str) -> str | None:
        course = self.find_course(code)
        if not course:
            return None

        with self.connect() as db:
            schedules = db.execute(
                """
                SELECT day,start_time,end_time,room,semester
                FROM schedules WHERE course_id=? ORDER BY id
                """,
                (course["id"],),
            ).fetchall()
            prerequisites = db.execute(
                """
                SELECT p.code,p.name_ar
                FROM prerequisites pr
                JOIN courses p ON p.id=pr.prerequisite_course_id
                WHERE pr.course_id=?
                """,
                (course["id"],),
            ).fetchall()

        lines = [
            f"📘 {course['code']} — {course['name_ar']}",
            f"الساعات المعتمدة: {course['credit_hours'] or 'غير محدد'}",
        ]
        if course["description"]:
            lines.append(f"الوصف: {course['description']}")
        if course["instructor_name"]:
            lines.append(f"المحاضر: {course['instructor_name']}")
        if schedules:
            lines.append("المواعيد:")
            for row in schedules:
                end = f"–{row['end_time']}" if row["end_time"] else ""
                room = f"، القاعة {row['room']}" if row["room"] else ""
                lines.append(f"• {row['day']} {row['start_time']}{end}{room}")
        if prerequisites:
            lines.append(
                "المتطلبات السابقة: "
                + "، ".join(f"{r['code']} ({r['name_ar']})" for r in prerequisites)
            )
        else:
            lines.append("المتطلبات السابقة: لا توجد بيانات مسجلة.")
        return "\n".join(lines)

    # -----------------------------------------------------
    # حالة الحوار وفهم الاستعلام
    # -----------------------------------------------------

    def get_state(self, session_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM conversation_state WHERE session_id=?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else {}

    def update_state(
        self,
        session_id: str,
        *,
        course_code: str | None = None,
        intent: str | None = None,
        pending_clarification: str | None = None,
    ) -> None:
        state = self.get_state(session_id)
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO conversation_state(
                    session_id,last_course_code,last_intent,
                    pending_clarification,updated_at
                ) VALUES(?,?,?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_course_code=excluded.last_course_code,
                    last_intent=excluded.last_intent,
                    pending_clarification=excluded.pending_clarification,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    session_id,
                    course_code if course_code is not None
                    else state.get("last_course_code"),
                    intent if intent is not None else state.get("last_intent"),
                    pending_clarification,
                ),
            )

    def detect_course_reference(
        self,
        query: str,
        state: dict[str, Any],
    ) -> tuple[str | None, bool]:
        match = self.COURSE_CODE_PATTERN.search(query)
        if match:
            code = f"{match.group(1)}{match.group(2)}".upper()
            if self.find_course(code):
                return code, False

        for course in self.list_courses():
            if course["name_ar"] and course["name_ar"] in query:
                return course["code"], False
            if course["name_en"] and course["name_en"].lower() in query.lower():
                return course["code"], False

        pronouns = ("هو", "هي", "هذا المساق", "المساق", "المادة", "عنه", "لها", "له")
        if any(token in query for token in pronouns) and state.get("last_course_code"):
            return str(state["last_course_code"]), True

        return None, False

    def detect_intent(self, query: str) -> str:
        q = query.lower()
        if any(word in q for word in ("موعد", "متى", "الساعة", "جدول", "محاضرة")):
            return "schedule"
        if any(word in q for word in ("قاعة", "أين", "مكان")):
            return "room"
        if any(word in q for word in ("متطلب", "متطلبات", "سابق", "قبل")):
            return "prerequisite"
        if any(word in q for word in ("أستاذ", "دكتور", "محاضر", "مدرس", "ساعات مكتبية")):
            return "instructor"
        if any(word in q for word in ("وصف", "معلومات", "تفاصيل", "ساعات معتمدة")):
            return "course_info"
        return "text_qa"

    def rewrite_query(
        self,
        original_query: str,
        course_code: str | None,
        used_context: bool,
    ) -> str:
        if course_code and used_context:
            return f"{original_query} المقصود هو مساق {course_code}."
        return original_query

    def structured_answer(self, intent: str, course_code: str) -> str | None:
        course = self.find_course(course_code)
        if not course:
            return None

        if intent == "course_info":
            return self.course_details(course_code)

        with self.connect() as db:
            if intent in {"schedule", "room"}:
                rows = db.execute(
                    """
                    SELECT day,start_time,end_time,room,semester
                    FROM schedules WHERE course_id=? ORDER BY id
                    """,
                    (course["id"],),
                ).fetchall()
                if not rows:
                    return "لا توجد مواعيد مسجلة لهذا المساق."
                heading = f"📅 مواعيد {course['code']} — {course['name_ar']}:"
                lines = [heading]
                for row in rows:
                    end = f"–{row['end_time']}" if row["end_time"] else ""
                    room = f"، القاعة {row['room']}" if row["room"] else ""
                    lines.append(f"• {row['day']} {row['start_time']}{end}{room}")
                return "\n".join(lines)

            if intent == "instructor":
                parts = [f"👨‍🏫 محاضر {course['code']}: {course['instructor_name'] or 'غير مسجل'}"]
                if course["email"]:
                    parts.append(f"البريد: {course['email']}")
                if course["office"]:
                    parts.append(f"المكتب: {course['office']}")
                if course["office_hours"]:
                    parts.append(f"الساعات المكتبية: {course['office_hours']}")
                return "\n".join(parts)

            if intent == "prerequisite":
                rows = db.execute(
                    """
                    SELECT p.code,p.name_ar
                    FROM prerequisites pr
                    JOIN courses p ON p.id=pr.prerequisite_course_id
                    WHERE pr.course_id=?
                    """,
                    (course["id"],),
                ).fetchall()
                if not rows:
                    return f"لا توجد متطلبات سابقة مسجلة لمساق {course['code']}."
                return (
                    f"🔗 متطلبات {course['code']}:\n"
                    + "\n".join(f"• {r['code']} — {r['name_ar']}" for r in rows)
                )
        return None

    async def process_query(
        self,
        session_id: str,
        query: str,
    ) -> QueryDecision:
        state = self.get_state(session_id)
        intent = self.detect_intent(query)
        course_code, used_context = self.detect_course_reference(query, state)
        rewritten = self.rewrite_query(query, course_code, used_context)

        structured_intents = {
            "schedule", "room", "prerequisite", "instructor", "course_info"
        }

        if intent in structured_intents and not course_code:
            courses = self.list_courses()
            if len(courses) == 1:
                course_code = courses[0]["code"]
            else:
                names = "، ".join(
                    f"{row['code']} ({row['name_ar']})"
                    for row in courses[:6]
                )
                clarification = (
                    "سؤالك يحتاج تحديد المساق. أي مساق تقصد؟\n"
                    f"المساقات المتاحة: {names}"
                )
                self.update_state(
                    session_id,
                    intent=intent,
                    pending_clarification=query,
                )
                return QueryDecision(
                    original_query=query,
                    rewritten_query=rewritten,
                    intent=intent,
                    route="clarification",
                    clarification_question=clarification,
                )

        if course_code:
            self.update_state(
                session_id,
                course_code=course_code,
                intent=intent,
                pending_clarification=None,
            )

        if intent in structured_intents and course_code:
            answer = self.structured_answer(intent, course_code)
            if answer:
                return QueryDecision(
                    original_query=query,
                    rewritten_query=rewritten,
                    intent=intent,
                    route="sql",
                    structured_answer=answer,
                    course_code=course_code,
                )

        return QueryDecision(
            original_query=query,
            rewritten_query=rewritten,
            intent=intent,
            route="rag",
            course_code=course_code,
        )

    # -----------------------------------------------------
    # التسجيل والتقييم
    # -----------------------------------------------------

    def log_interaction(
        self,
        session_id: str,
        decision: QueryDecision,
        *,
        response_time_ms: float,
        retrieved_count: int,
        answer: str,
    ) -> int:
        with self.connect() as db:
            cursor = db.execute(
                """
                INSERT INTO research_logs(
                    session_id,original_query,rewritten_query,intent,route,
                    course_code,clarification_used,retrieved_count,
                    response_time_ms,answer_preview
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    session_id,
                    decision.original_query,
                    decision.rewritten_query,
                    decision.intent,
                    decision.route,
                    decision.course_code,
                    1 if decision.route == "clarification" else 0,
                    retrieved_count,
                    response_time_ms,
                    answer[:1000],
                ),
            )
            return int(cursor.lastrowid)

    def log_deep_retrieval(
        self,
        *,
        research_log_id: int | None,
        session_id: str,
        query_variants: list[str],
        stages: int,
        candidate_count: int,
        evidence_count: int,
        sufficient: bool,
        confidence: float,
        missing_evidence: str,
    ) -> None:
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO deep_retrieval_logs(
                    research_log_id,session_id,query_variants,stages,
                    candidate_count,evidence_count,sufficient,confidence,
                    missing_evidence
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    research_log_id,
                    session_id,
                    json.dumps(query_variants, ensure_ascii=False),
                    stages,
                    candidate_count,
                    evidence_count,
                    1 if sufficient else 0,
                    confidence,
                    missing_evidence,
                ),
            )

    def latest_log_id(self, session_id: str) -> int | None:
        with self.connect() as db:
            row = db.execute(
                """
                SELECT id FROM research_logs
                WHERE session_id=? ORDER BY id DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return int(row["id"]) if row else None

    def add_feedback(
        self,
        session_id: str,
        rating: int,
        comment: str = "",
    ) -> None:
        log_id = self.latest_log_id(session_id)
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO feedback(log_id,session_id,rating,comment)
                VALUES(?,?,?,?)
                """,
                (log_id, session_id, rating, comment),
            )

    def research_stats(self) -> dict[str, Any]:
        with self.connect() as db:
            total = db.execute("SELECT COUNT(*) FROM research_logs").fetchone()[0]
            avg_time = db.execute(
                "SELECT AVG(response_time_ms) FROM research_logs"
            ).fetchone()[0] or 0
            clarification = db.execute(
                "SELECT COUNT(*) FROM research_logs WHERE clarification_used=1"
            ).fetchone()[0]
            avg_rating = db.execute(
                "SELECT AVG(rating) FROM feedback"
            ).fetchone()[0] or 0
            routes = db.execute(
                """
                SELECT route,COUNT(*) AS count
                FROM research_logs GROUP BY route ORDER BY count DESC
                """
            ).fetchall()
            deep_count = db.execute(
                "SELECT COUNT(*) FROM deep_retrieval_logs"
            ).fetchone()[0]
            deep_confidence = db.execute(
                "SELECT AVG(confidence) FROM deep_retrieval_logs"
            ).fetchone()[0] or 0
            deep_sufficient = db.execute(
                """
                SELECT AVG(CAST(sufficient AS REAL))
                FROM deep_retrieval_logs
                """
            ).fetchone()[0] or 0

        return {
            "total_queries": total,
            "average_response_ms": round(float(avg_time), 2),
            "clarification_count": clarification,
            "average_rating": round(float(avg_rating), 2),
            "routes": {row["route"]: row["count"] for row in routes},
            "deep_query_count": deep_count,
            "deep_average_confidence": round(float(deep_confidence), 3),
            "deep_sufficiency_rate": round(float(deep_sufficient) * 100, 2),
        }

    def export_logs_csv(self, output_path: Path) -> int:
        import csv
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT
                    l.*,
                    f.rating,
                    f.comment,
                    d.query_variants,
                    d.stages AS deep_stages,
                    d.candidate_count AS deep_candidate_count,
                    d.evidence_count AS deep_evidence_count,
                    d.sufficient AS deep_sufficient,
                    d.confidence AS deep_confidence,
                    d.missing_evidence AS deep_missing_evidence
                FROM research_logs l
                LEFT JOIN feedback f ON f.log_id=l.id
                LEFT JOIN deep_retrieval_logs d
                    ON d.research_log_id=l.id
                ORDER BY l.id
                """
            ).fetchall()

        with output_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(rows[0].keys() if rows else [
                "id","session_id","original_query","rewritten_query","intent",
                "route","course_code","clarification_used","retrieved_count",
                "response_time_ms","answer_preview","created_at","rating","comment"
            ])
            for row in rows:
                writer.writerow(list(row))
        return len(rows)
