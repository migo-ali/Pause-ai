import unittest
from school_management import School, Student, Teacher, Course, Enrollment


class SchoolManagementTest(unittest.TestCase):
    def test_basic_flow(self):
        school = School()
        school.add_student(Student(1, "Alice", "10"))
        school.add_teacher(Teacher(1, "Mr. Smith", "Math"))
        school.add_course(Course(1, "Algebra", 1))
        school.enroll_student(Enrollment(1, 1))

        self.assertEqual(len(school.list_students()), 1)
        self.assertEqual(len(school.courses), 1)
        self.assertEqual(len(school.list_enrollments()), 1)


if __name__ == "__main__":
    unittest.main()
