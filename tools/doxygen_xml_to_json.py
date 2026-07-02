#!/usr/bin/env python3
"""Convert Doxygen XML output into static JSON files for web UIs.

The generated layout is optimized for a frontend with one reusable page:

  api/index.json                Lightweight browse index
  api/search.json               Flat search index
  api/compounds/<refid>.json    One detailed JSON document per Doxygen XML compound

This avoids loading the whole documentation set up front while keeping every
resource static and GitHub Pages friendly.
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


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    result: dict[str, Any] = {
        "id": member.get("id", ""),
        "name": child_text(member, "name"),
        "kind": member.get("kind", ""),
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


def parse_compound(xml_dir: Path, refid: str, fallback_kind: str, fallback_name: str) -> dict[str, Any]:
    path = xml_dir / f"{refid}.xml"
    if not path.exists():
        return {
            "id": refid,
            "kind": fallback_kind,
            "name": fallback_name,
            "brief": "",
            "description": "",
            "location": {},
            "members": [],
            "inner": [],
            "includes": [],
            "source": {"xmlPath": f"xml/{refid}.xml"},
        }

    root = read_xml(path)
    compound = root.find("compounddef")
    if compound is None:
        raise RuntimeError(f"No compounddef found in {path}")

    compound_name = child_text(compound, "compoundname") or fallback_name
    result: dict[str, Any] = {
        "id": refid,
        "kind": compound.get("kind", "") or fallback_kind,
        "name": compound_name,
        "language": compound.get("language", ""),
        "brief": node_text(compound.find("briefdescription")),
        "description": node_text(compound.find("detaileddescription")),
        "location": location(compound),
        "members": [],
        "inner": [],
        "includes": [],
        "source": {"xmlPath": f"xml/{refid}.xml"},
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
                "path": f"api/compounds/{inner.get('refid', '')}.json" if inner.get("refid") else "",
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


def compound_summary(compound: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": compound["id"],
        "kind": compound.get("kind", ""),
        "name": compound.get("name", ""),
        "brief": compound.get("brief", ""),
        "location": compound.get("location", {}),
        "memberCount": len(compound.get("members", [])),
        "path": f"api/compounds/{compound['id']}.json",
        "xmlPath": compound.get("source", {}).get("xmlPath", f"xml/{compound['id']}.xml"),
    }


def search_entries_for(compound: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = [
        {
            "id": compound["id"],
            "kind": compound.get("kind", ""),
            "name": compound.get("name", ""),
            "brief": compound.get("brief", ""),
            "parentId": "",
            "parentName": "",
            "location": compound.get("location", {}),
            "path": f"api/compounds/{compound['id']}.json",
        }
    ]

    for member in compound.get("members", []):
        entries.append(
            {
                "id": member.get("id", ""),
                "kind": member.get("kind", ""),
                "name": member.get("name", ""),
                "signature": member.get("signature", ""),
                "brief": member.get("brief", ""),
                "parentId": compound["id"],
                "parentName": compound.get("name", ""),
                "location": member.get("location", {}),
                "path": f"api/compounds/{compound['id']}.json#{member.get('id', '')}",
            }
        )

    return entries


def build_site_json(xml_dir: Path, out_dir: Path, project: str, repo: str) -> dict[str, Any]:
    index_path = xml_dir / "index.xml"
    if not index_path.exists():
        raise FileNotFoundError(f"Doxygen index not found: {index_path}")

    generated_at = datetime.now(timezone.utc).isoformat()
    root = read_xml(index_path)

    compounds: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    search: list[dict[str, Any]] = []
    stats: dict[str, int] = {"compounds": 0, "searchItems": 0}

    for compound_node in root.findall("compound"):
        refid = compound_node.get("refid", "")
        if not refid:
            continue

        compound = parse_compound(
            xml_dir=xml_dir,
            refid=refid,
            fallback_kind=compound_node.get("kind", ""),
            fallback_name=child_text(compound_node, "name"),
        )

        compound_payload = {
            "schemaVersion": 1,
            "project": project,
            "repository": repo,
            "generatedAt": generated_at,
            "compound": compound,
        }
        write_json(out_dir / "compounds" / f"{refid}.json", compound_payload)

        compounds.append(compound)
        summaries.append(compound_summary(compound))
        search.extend(search_entries_for(compound))

        kind_key = f"{compound.get('kind', 'unknown')}Count"
        stats[kind_key] = stats.get(kind_key, 0) + 1

    summaries.sort(key=lambda item: (item.get("kind", ""), item.get("name", "")))
    search.sort(key=lambda item: (item.get("name", ""), item.get("kind", ""), item.get("parentName", "")))

    stats["compounds"] = len(summaries)
    stats["searchItems"] = len(search)

    index_payload = {
        "schemaVersion": 1,
        "project": project,
        "repository": repo,
        "generatedAt": generated_at,
        "source": {"format": "doxygen-xml", "xmlPath": "xml/index.xml"},
        "paths": {
            "index": "api/index.json",
            "search": "api/search.json",
            "compoundsBase": "api/compounds/",
            "rawXmlBase": "xml/",
        },
        "stats": stats,
        "compounds": summaries,
    }

    search_payload = {
        "schemaVersion": 1,
        "project": project,
        "repository": repo,
        "generatedAt": generated_at,
        "stats": stats,
        "items": search,
    }

    write_json(out_dir / "index.json", index_payload)
    write_json(out_dir / "search.json", search_payload)

    return index_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Doxygen XML into static JSON files for web consumers.")
    parser.add_argument("--xml-dir", type=Path, required=True, help="Directory containing Doxygen XML output")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for generated JSON files")
    parser.add_argument("--project", default="arc", help="Project name for the payload")
    parser.add_argument("--repo", default="richardthorell/arc", help="Repository name for the payload")
    args = parser.parse_args()

    payload = build_site_json(args.xml_dir, args.out_dir, args.project, args.repo)
    print(
        f"Wrote docs JSON to {args.out_dir} with {payload['stats']['compounds']} compounds "
        f"and {payload['stats']['searchItems']} search items"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
