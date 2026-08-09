import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import tifffile
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tensorspec.core.workspace import WorkspaceManager
from tensorspec.web.server.app import create_app
from tensorspec.web.server.routers import peem as peem_router
from tensorspec.web.server.session import Session, current_session


class TestPeemApi(unittest.TestCase):
    def _session(self, tmp: str) -> Session:
        return Session(session_id="t", workspace=WorkspaceManager(project_dir=Path(tmp)))

    def test_load_server_path_with_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run"
            root.mkdir()
            tifffile.imwrite(root / "a_CP.tif", np.ones((4, 4), dtype=np.uint16))
            tifffile.imwrite(root / "b_CM.tif", np.full((4, 4), 2, dtype=np.uint16))
            (root / "run.csv").write_text("I0\n1.1\n1.2\n")

            summary = peem_router.load_peem(
                file=None,
                server_path=str(root),
                csv=None,
                csv_path=None,
                name="peem1",
                session=session,
            )

            self.assertEqual(summary.n_frames, 2)
            self.assertTrue(summary.csv_attached)
            self.assertTrue(summary.I0_present)
            self.assertEqual(summary.pol_summary, {"CP": 1, "CM": 1})
            meta = peem_router.get_meta("peem1", session=session)
            self.assertEqual(meta.pol, ["CP", "CM"])
            frame = peem_router.get_frame("peem1", 0, session=session)
            self.assertEqual(frame.shape, [4, 4])
            self.assertEqual(len(frame.intensity), 4)
            self.assertEqual((frame.vmin, frame.vmax), (1.0, 1.0))

    def test_attach_csv_after_load_merges_raw_attrs(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run"
            root.mkdir()
            tifffile.imwrite(root / "a.tif", np.ones((2, 2), dtype=np.uint16))
            summary = peem_router.load_peem(
                file=None,
                server_path=str(root),
                csv=None,
                csv_path=None,
                name="peem2",
                session=session,
            )
            self.assertFalse(summary.csv_attached)
            csv_path = root / "late.csv"
            csv_path.write_text("I0\n3.3\n")

            peem_router.attach_csv(
                "peem2", csv=None, csv_path=str(csv_path), session=session
            )

            meta = peem_router.get_meta("peem2", session=session)
            self.assertTrue(meta.csv_attached)
            self.assertTrue(meta.I0_present)
            self.assertEqual(meta.I0, 3.3)
            tensor = session.workspace.pull_tensor_data("peem2")
            self.assertEqual(tensor.metadata["beamline_csv"], str(csv_path.resolve()))
            self.assertEqual(tensor.metadata["beamline_table"]["series"]["I0"], [3.3])

    def test_replacing_csv_clears_stale_beam_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run"
            root.mkdir()
            tifffile.imwrite(root / "a.tif", np.ones((2, 2), dtype=np.uint16))
            peem_router.load_peem(
                file=None,
                server_path=str(root),
                csv=None,
                csv_path=None,
                name="replace",
                session=session,
            )
            first = root / "first.csv"
            first.write_text("beam_current\n8.1\n")
            second = root / "second.csv"
            second.write_text("I0\n2.4\n")
            peem_router.attach_csv(
                "replace", csv=None, csv_path=str(first), session=session
            )
            self.assertEqual(
                session.workspace.pull_tensor_data("replace").metadata["beam_current"],
                8.1,
            )

            peem_router.attach_csv(
                "replace", csv=None, csv_path=str(second), session=session
            )

            metadata = session.workspace.pull_tensor_data("replace").metadata
            self.assertIsNone(metadata.get("beam_current"))
            self.assertEqual(metadata["I0"], 2.4)
            self.assertEqual(metadata["beamline_csv"], str(second.resolve()))

    def test_ambiguous_csv_requests_prompt_with_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run"
            root.mkdir()
            tifffile.imwrite(root / "a.tif", np.ones((2, 2), dtype=np.uint16))
            one = root / "one.csv"
            two = root / "two.csv"
            one.write_text("I0\n1\n")
            two.write_text("I0\n2\n")

            summary = peem_router.load_peem(
                file=None,
                server_path=str(root),
                csv=None,
                csv_path=None,
                name="ambiguous",
                session=session,
            )

            self.assertFalse(summary.csv_attached)
            self.assertTrue(summary.csv_prompt)
            self.assertEqual(
                summary.csv_candidates, [str(one.resolve()), str(two.resolve())]
            )

    def test_auto_discovered_csv_enforces_size_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run"
            root.mkdir()
            tifffile.imwrite(root / "a.tif", np.ones((2, 2), dtype=np.uint16))
            oversized = root / "run.csv"
            oversized.write_bytes(b"I0\n" + b"x" * peem_router.MAX_CSV_BYTES)

            with self.assertRaises(HTTPException) as ctx:
                peem_router.load_peem(
                    file=None,
                    server_path=str(root),
                    csv=None,
                    csv_path=None,
                    name="oversized",
                    session=session,
                )

            self.assertEqual(ctx.exception.status_code, 413)

    def test_path_escape_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            with self.assertRaises(HTTPException) as ctx:
                peem_router.load_peem(
                    file=None,
                    server_path="/etc/passwd",
                    csv=None,
                    csv_path=None,
                    name="x",
                    session=session,
                )
            self.assertEqual(ctx.exception.status_code, 403)

    def test_load_requires_exactly_one_image_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            with self.assertRaises(HTTPException) as ctx:
                peem_router.load_peem(
                    file=None,
                    server_path=None,
                    csv=None,
                    csv_path=None,
                    name="x",
                    session=session,
                )
            self.assertEqual(ctx.exception.status_code, 400)

    def test_frame_index_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run"
            root.mkdir()
            tifffile.imwrite(root / "a.tif", np.ones((2, 2), dtype=np.uint16))
            peem_router.load_peem(
                file=None,
                server_path=str(root),
                csv=None,
                csv_path=None,
                name="peem",
                session=session,
            )
            with self.assertRaises(HTTPException) as ctx:
                peem_router.get_frame("peem", 1, session=session)
            self.assertEqual(ctx.exception.status_code, 404)

    def test_zip_upload_route_loads_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            source = Path(tmp) / "source"
            source.mkdir()
            tifffile.imwrite(source / "a_CP.tif", np.ones((2, 3), dtype=np.uint16))
            tifffile.imwrite(source / "b_CM.tif", np.full((2, 3), 2, dtype=np.uint16))
            payload = io.BytesIO()
            with zipfile.ZipFile(payload, "w") as archive:
                archive.write(source / "a_CP.tif", "nested/a_CP.tif")
                archive.write(source / "b_CM.tif", "nested/b_CM.tif")

            app = create_app()
            app.dependency_overrides[current_session] = lambda: session
            client = TestClient(app)
            response = client.post(
                "/api/peem/load",
                files={"file": ("run.zip", payload.getvalue(), "application/zip")},
                data={"name": "uploaded"},
            )

            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["shape"], [2, 2, 3])
            self.assertEqual(response.json()["pol_summary"], {"CP": 1, "CM": 1})
            frame = client.get("/api/peem/uploaded/frame/1")
            self.assertEqual(frame.status_code, 200, frame.text)
            self.assertEqual(frame.json()["shape"], [2, 3])
            self.assertEqual(frame.json()["pol"], "CM")
            self.assertEqual(frame.json()["intensity"], [[2.0] * 3, [2.0] * 3])

    def test_repeated_upload_filename_uses_clean_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            app = create_app()
            app.dependency_overrides[current_session] = lambda: session
            client = TestClient(app)

            first = io.BytesIO()
            with zipfile.ZipFile(first, "w") as archive:
                for filename, value in (("a.tif", 1), ("stale.tif", 9)):
                    frame = io.BytesIO()
                    tifffile.imwrite(frame, np.full((2, 2), value, dtype=np.uint16))
                    archive.writestr(filename, frame.getvalue())
            response = client.post(
                "/api/peem/load",
                files={"file": ("same.zip", first.getvalue(), "application/zip")},
                data={"name": "first"},
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["n_frames"], 2)

            second = io.BytesIO()
            with zipfile.ZipFile(second, "w") as archive:
                frame = io.BytesIO()
                tifffile.imwrite(frame, np.full((2, 2), 3, dtype=np.uint16))
                archive.writestr("a.tif", frame.getvalue())
            response = client.post(
                "/api/peem/load",
                files={"file": ("same.zip", second.getvalue(), "application/zip")},
                data={"name": "second"},
            )

            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["n_frames"], 1)
            self.assertNotEqual(
                Path(session.workspace.pull_tensor_data("first").metadata["source"]),
                Path(session.workspace.pull_tensor_data("second").metadata["source"]),
            )


if __name__ == "__main__":
    unittest.main()
