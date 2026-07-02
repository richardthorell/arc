#!/usr/bin/env python3
"""Convert Doxygen XML output into a compact JSON index for web UIs.

The output is intentionally static-site friendly: a single JSON file with enough
structure for search, browse, and detail pages without needing a backend.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return WHITESPACE_RE.sub(" ", value).strip()


def node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return clean_text("".join(node.itertext()))


def child_text(node: ET.Element, name: str) -> str:
    return node_text(node.find(name))


def location(node: ET.Element) -> dict[str, Any]:
    loc = node.find("location")
    if loc is None:
        return {}

    result: dict[str, Any] = {}
    for key in ("file", "line", "column", "bodyfile", "bodystart", "bodyend"):
        value = loc.get(key)
        if value:
            if key in {"line", "column", "bodystart", "bodyend"}:
                try:
                    result[key] = int(value)
                except ValueError:
                    result[key] = value
            else:
                result[key] = value
    return result


def read_xml(path: Path) -> ET.Element:
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise RuntimeError(f"Failed to parse {path}: {exc}") from exc


def member_signature(member: ET.Element) -> str:
    definition = child_text(member, "definition")
    args = child_text(member, "argsstring")
    signature = clean_text(f"{definition}{args}")
    return signature or child_text(member, "name")


def parse_member(member: ET.Element, parent_refid: str, parent_name: str) -> dict[str, Any]:
    name = child_text(member, "name")
    kind = member.get("kind", "")
    refid = member.get("id", "")

    result: dict[str, Any] = {
        "id": refid,
        "name": name,
        "kind": kind,
        "parentId": parent_refid,
        "parentName": parent_name,
        "visibility": member.get("prot", ""),
        "static": member.get("static") == "yes",
        "const": member.get("const") == "yes",
        "type": node_text(member.find("type")),
        "signature": member_signature(member),
        "brief": node_text(member.find("briefdescription")),
        "description": node_text(member.find("detaileddescription")),
        "location": location(member),
    }

    params = []
    for param in member.findall("param"):
        params.append(
            {
                "type": node_text(param.find("type")),
                "declname": child_text(param, "declname"),
                "defname": child_text(param, "defname"),
                "array": child_text(param, "array"),
                "default": node_text(param.find("defval")),
            }
        )
    if params:
        result["params"] = params

    enum_values = []
    for enum_value in member.findall("enumvalue"):
        enum_values.append(
            {
                "id": enum_value.get("id", ""),
                "name": child_text(enum_value, "name"),
                "brief": node_text(enum_value.find("briefdescription")),
                "description": node_text(enum_value.find("detaileddescription")),
            }
        )
    if enum_values:
        result["values"] = enum_values

    return result


def parse_compound(xml_dir: Path, refid: str) -> dict[str, Any] | None:
    path = xml_dir / f"{refid}.xml"
    if not path.exists():
        return None

    root = read_xml(path)
    compound = root.find("compounddef")
    if compound is None:
        return None

    compound_name = child_text(compound, "compoundname")
    result: dict[str, Any] = {
        "id": refid,
        "kind": compound.get("kind", ""),
        "name": compound_name,
        "language": compound.get("language", ""),
        "brief": node_text(compound.find("briefdescription")),
        "description": node_text(compound.find("detaileddescription")),
        "location": location(compound),
        "members": [],
        "inner": [],
        "includes": [],
    }

    for include in compound.findall("includes"):
        text = node_text(include)
        if text:
            result["includes"].append(
                {
                    "name": text,
                    "refid": include.get("refid", ""),
                    "local": include.get("local", ""),
                }
            )

    for inner in list(compound.findall("innerclass")) + list(compound.findall("innernamespace")) + list(compound.findall("innerfile")):
        result["inner"].append(
            {
                "id": inner.get("refid", ""),
                "kind": inner.tag.removeprefix("inner"),
                "name": node_text(inner),
            }
        )

    members: list[dict[str, Any]] = []
    for section in compound.findall("sectiondef"):
        section_kind = section.get("kind", "")
        for member in section.findall("memberdef"):
            parsed = parse_member(member, refid, compound_name)
            parsed["section"] = section_kind
            members.append(parsed)
    result["members"] = members

    return result


def parse_index(xml_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index_path = xml_dir / "index.xml"
    if not index_path.exists():
        raise FileNotFoundError(f"Doxygen index not found: {index_path}")

    root = read_xml(index_path)
    compounds: list[dict[str, Any]] = []
    search_items: list[dict[str, Any]] = []

    for compound_node in root.findall("compound"):
        refid = compound_node.get("refid", "")
        kind = compound_node.get("kind", "")
        name = child_text(compound_node, "name")
        parsed = parse_compound(xml_dir, refid)

        if parsed is None:
            parsed = {
                "id": refid,
                "kind": kind,
                "name": name,
                "members": [],
                "inner": [],
                "includes": [],
            }
        else:
            parsed["kind"] = parsed.get("kind") or kind
            parsed["name"] = parsed.get("name") or name

        compounds.append(parsed)
        search_items.append(
            {
                "id": parsed["id"],
                "kind": parsed["kind"],
                "name": parsed["name"],
                "brief": parsed.get("brief", ""),
                "parentId": "",
                "parentName": "",
                "location": parsed.get("location", {}),
            }
        )

        for member in parsed.get("members", []):
            search_items.append(
                {
                    "id": member.get("id", ""),
                    "kind": member.get("kind", ""),
                    "name": member.get("name", ""),
                    "signature": member.get("signature", ""),
                    "brief": member.get("brief", ""),
                    "parentId": parsed["id"],
                    "parentName": parsed["name"],
                    "location": member.get("location", {}),
                }
            )

    compounds.sort(key=lambda item: (item.get("kind", ""), item.get("name", "")))
    search_items.sort(key=lambda item: (item.get("name", ""), item.get("kind", ""), item.get("parentName", "")))
    return compounds, search_items


def build_payload(xml_dir: Path, project: str, repo: str) -> dict[str, Any]:
    compounds, search_items = parse_index(xml_dir)

    stats: dict[str, int] = {
        "compounds": len(compounds),
        "searchItems": len(search_items),
    }
    for compound in compounds:
        key = f"{compound.get('kind', 'unknown')}Count"
        stats[key] = stats.get(key, 0) + 1

    return {
        "schemaVersion": 1,
        "project": project,
        "repository": repo,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": {
            "format": "doxygen-xml",
            "xmlPath": "xml/index.xml",
        },
        "stats": stats,
        "compounds": compounds,
        "search": search_items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Doxygen XML into JSON for static web consumers.")
    parser.add_argument("--xml-dir", type=Path, required=True, help="Directory containing Doxygen XML output")
    parser.add_argument("--out", type=Path, required=True, help="JSON file to write")
    parser.add_argument("--project", default="arc", help="Project name for the payload")
    parser.add_argument("--repo", default="richardthorell/arc", help="Repository name for the payload")
    args = parser.parse_args()

    payload = build_payload(args.xml_dir, args.project, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"Wrote {args.out} with {payload['stats']['compounds']} compounds "
        f"and {payload['stats']['searchItems']} search items"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
