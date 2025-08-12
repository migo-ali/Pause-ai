from __future__ import annotations

"""Simple school management program.

This module provides a basic in-memory management system for students,
teachers, courses, and enrollments. Data can be persisted to and from a JSON
file. A small command-line interface is included for demonstration.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List
import json
import argparse
import sys


@dataclass
class Student:
    """Represents a student in the school."""
    student_id: int
    name: str
    grade: str


@dataclass
class Teacher:
    """Represents a teacher in the school."""
    teacher_id: int
    name: str
    subject: str


@dataclass
class Course:
    """Represents a course taught by a teacher."""
    course_id: int
    name: str
    teacher_id: int


@dataclass
class Enrollment:
    """Links a student to a course."""
    student_id: int
    course_id: int


class School:
    """A simple in-memory school management system."""

    def __init__(self) -> None:
        self.students: Dict[int, Student] = {}
        self.teachers: Dict[int, Teacher] = {}
        self.courses: Dict[int, Course] = {}
        self.enrollments: List[Enrollment] = []

    # Student management
    def add_student(self, student: Student) -> None:
        if student.student_id in self.students:
            raise ValueError("Student ID already exists")
        self.students[student.student_id] = student

    def list_students(self) -> List[Student]:
        return list(self.students.values())

    # Teacher management
    def add_teacher(self, teacher: Teacher) -> None:
        if teacher.teacher_id in self.teachers:
            raise ValueError("Teacher ID already exists")
        self.teachers[teacher.teacher_id] = teacher

    # Course management
    def add_course(self, course: Course) -> None:
        if course.course_id in self.courses:
            raise ValueError("Course ID already exists")
        if course.teacher_id not in self.teachers:
            raise ValueError("Teacher ID does not exist")
        self.courses[course.course_id] = course

    # Enrollment management
    def enroll_student(self, enrollment: Enrollment) -> None:
        if enrollment.student_id not in self.students:
            raise ValueError("Student ID does not exist")
        if enrollment.course_id not in self.courses:
            raise ValueError("Course ID does not exist")
        self.enrollments.append(enrollment)

    def list_enrollments(self) -> List[Enrollment]:
        return list(self.enrollments)

    # Persistence
    def save(self, path: str) -> None:
        data = {
            "students": [asdict(s) for s in self.students.values()],
            "teachers": [asdict(t) for t in self.teachers.values()],
            "courses": [asdict(c) for c in self.courses.values()],
            "enrollments": [asdict(e) for e in self.enrollments],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "School":
        school = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("students", []):
            school.students[s["student_id"]] = Student(**s)
        for t in data.get("teachers", []):
            school.teachers[t["teacher_id"]] = Teacher(**t)
        for c in data.get("courses", []):
            school.courses[c["course_id"]] = Course(**c)
        for e in data.get("enrollments", []):
            school.enrollments.append(Enrollment(**e))
        return school


# Command-line interface -----------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="School management CLI")
    sub = parser.add_subparsers(dest="command")

    add_student_p = sub.add_parser("add-student", help="Add a new student")
    add_student_p.add_argument("student_id", type=int)
    add_student_p.add_argument("name")
    add_student_p.add_argument("grade")

    list_students_p = sub.add_parser("list-students", help="List all students")

    return parser

def main(argv: List[str]) -> int:
    school = School()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "add-student":
        school.add_student(Student(args.student_id, args.name, args.grade))
        print("Student added")
    elif args.command == "list-students":
        for s in school.list_students():
            print(f"{s.student_id}: {s.name} ({s.grade})")
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
