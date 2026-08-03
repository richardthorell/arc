#!/usr/bin/env python3
import pathlib
import subprocess
import sys
import unittest


GENERATOR = pathlib.Path(sys.argv.pop(1)).resolve()
WORK_ROOT = pathlib.Path(sys.argv.pop(1)).resolve()


class ReflectionGeneratorTests(unittest.TestCase):
    sequence = 0

    def generate(self, source: str):
        ReflectionGeneratorTests.sequence += 1
        prefix = f"case-{ReflectionGeneratorTests.sequence}"
        header = WORK_ROOT / f"{prefix}-Component.h"
        header.write_text(source, encoding="utf-8")
        output = WORK_ROOT / f"{prefix}-Generated.h"
        schema = WORK_ROOT / f"{prefix}-schema.json"
        result = subprocess.run(
            [
                sys.executable,
                str(GENERATOR),
                "--header",
                str(header),
                "--cpp-output",
                str(output),
                "--json-output",
                str(schema),
                "--namespace",
                "fixture::generated",
                "--component-namespace",
                "fixture",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return result, output, schema

    def test_explicit_ids_survive_cpp_renames(self):
        before = """
#include <arc/project/reflection.h>
namespace fixture {
ARC_COMPONENT("1234567890abcdef1234567890abcdef", 1, "Stats", "Test", "Description")
struct old_name {
 ARC_PROPERTY("0123456789abcdef", "Value", "Test", "Description", "float", "1.0", "", "",
              "editable|serialized", "", "")
 float old_field{1.0F};
}; }
"""
        after = before.replace("old_name", "new_name").replace("old_field", "new_field")
        first, first_output, first_schema = self.generate(before)
        second, second_output, second_schema = self.generate(after)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("1234567890abcdef1234567890abcdef", first_schema.read_text())
        self.assertIn("1234567890abcdef1234567890abcdef", second_schema.read_text())
        self.assertIn("0x0123456789abcdefull", first_output.read_text())
        self.assertIn("0x0123456789abcdefull", second_output.read_text())

    def test_duplicate_component_ids_are_rejected(self):
        source = """
#include <arc/project/reflection.h>
namespace fixture {
ARC_COMPONENT("1234567890abcdef1234567890abcdef", 1, "A", "Test", "A")
struct a {};
ARC_COMPONENT("1234567890abcdef1234567890abcdef", 1, "B", "Test", "B")
struct b {};
}
"""
        result, _, _ = self.generate(source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate component stable ID", result.stderr)


if __name__ == "__main__":
    unittest.main()
