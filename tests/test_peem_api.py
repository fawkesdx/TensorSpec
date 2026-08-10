import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import tifffile
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tensorspec.core.data_models import TensorData
from tensorspec.core.workspace import WorkspaceManager
from tensorspec.web.server.app import create_app
from tensorspec.web.server.routers import peem as peem_router
from tensorspec.web.server.schemas import PeemDriftRequest, PeemPairRequest, PeemRoi
from tensorspec.web.server.session import Session, current_session


class TestPeemApi(unittest.TestCase):
    def _session(self, tmp: str) -> Session:
        return Session(session_id="t", workspace=WorkspaceManager(project_dir=Path(tmp)))

    def test_drift_raw_writes_processed_3d(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run"
            root.mkdir()
            base = np.zeros((32, 32), dtype=np.uint16)
            base[10:15, 11:17] = 20
            shifted = np.zeros_like(base)
            shifted[:, 3:] = base[:, :-3]
            tifffile.imwrite(root / "a.tif", base)
            tifffile.imwrite(root / "b.tif", shifted)
            peem_router.load_peem(
                file=None,
                server_path=str(root),
                csv=None,
                csv_path=None,
                name="drift_raw",
                session=session,
            )
            app = create_app()
            app.dependency_overrides[current_session] = lambda: session
            client = TestClient(app)
            request = PeemDriftRequest(
                source="raw",
                ref_index=0,
                search_radius=6,
                roi=PeemRoi(kind="rect", x0=7, y0=7, x1=24, y1=22),
            )

            response = client.post(
                "/api/peem/drift_raw/drift", json=request.model_dump()
            )

            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["shape"], [2, 32, 32])
            self.assertTrue(response.json()["has_drift"])
            meta = client.get("/api/peem/drift_raw/meta")
            self.assertEqual(meta.status_code, 200, meta.text)
            self.assertTrue(meta.json()["has_drift"])
            self.assertEqual(meta.json()["drift_method"], "ncc_roi")
            frame = client.get("/api/peem/drift_raw/frame/0?node=processed")
            self.assertEqual(frame.status_code, 200, frame.text)
            self.assertEqual(frame.json()["shape"], [32, 32])
            self.assertIsNone(frame.json()["pair"])
            self.assertIsNone(frame.json()["channel"])

    def test_drift_paired_keeps_4d_and_same_channel_shift(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run"
            root.mkdir()
            base = np.zeros((32, 32), dtype=np.uint16)
            base[10:15, 11:17] = 20
            other = base + 5
            shifted_base = np.zeros_like(base)
            shifted_other = np.zeros_like(other)
            shifted_base[2:, :] = base[:-2, :]
            shifted_other[2:, :] = other[:-2, :]
            for filename, plane in (
                ("a_CP.tif", base),
                ("b_CM.tif", other),
                ("c_CP.tif", shifted_base),
                ("d_CM.tif", shifted_other),
            ):
                tifffile.imwrite(root / filename, plane)
            peem_router.load_peem(
                file=None,
                server_path=str(root),
                csv=None,
                csv_path=None,
                name="drift_pair",
                session=session,
            )
            app = create_app()
            app.dependency_overrides[current_session] = lambda: session
            client = TestClient(app)
            pair = client.post(
                "/api/peem/drift_pair/pair",
                json=PeemPairRequest(mode="CP_CM").model_dump(),
            )
            self.assertEqual(pair.status_code, 200, pair.text)

            response = client.post(
                "/api/peem/drift_pair/drift",
                json={
                    "source": "processed",
                    "ref_index": 0,
                    "search_radius": 5,
                    "track_channel": 0,
                    "roi": {
                        "kind": "rect",
                        "x0": 7,
                        "y0": 7,
                        "x1": 24,
                        "y1": 22,
                    },
                },
            )

            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["shape"], [2, 2, 32, 32])
            processed = session.workspace.pull_tensor_data("drift_pair", "processed")
            self.assertEqual(processed.value.shape, (2, 2, 32, 32))
            self.assertEqual(processed.metadata["drift_shifts"][1]["dy"], -2)
            np.testing.assert_array_equal(
                processed.value[1, 1, :-2], processed.value[0, 1, :-2]
            )
            meta = client.get("/api/peem/drift_pair/meta")
            self.assertTrue(meta.json()["has_drift"])
            frame = client.get(
                "/api/peem/drift_pair/frame/1?node=processed&channel=1"
            )
            self.assertEqual(frame.status_code, 200, frame.text)
            self.assertEqual(frame.json()["channel_tag"], "CM")

    def test_pair_writes_processed_and_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run"
            root.mkdir()
            tifffile.imwrite(root / "a_CP.tif", np.ones((4, 4), dtype=np.uint16))
            tifffile.imwrite(root / "b_CM.tif", np.full((4, 4), 2, dtype=np.uint16))
            peem_router.load_peem(
                file=None,
                server_path=str(root),
                csv=None,
                csv_path=None,
                name="pair_me",
                session=session,
            )

            summary = peem_router.pair_peem(
                "pair_me", PeemPairRequest(mode="CP_CM"), session=session
            )

            self.assertEqual(summary.n_pairs, 1)
            self.assertTrue(summary.has_processed)
            self.assertEqual(summary.shape, [1, 2, 4, 4])
            meta = peem_router.get_meta("pair_me", session=session)
            self.assertTrue(meta.has_processed)
            self.assertEqual(meta.n_pairs, 1)
            frame = peem_router.get_frame(
                "pair_me", 0, node="processed", channel=1, session=session
            )
            self.assertEqual(frame.shape, [4, 4])
            self.assertEqual(frame.channel_tag, "CM")
            self.assertEqual(frame.node, "processed")
            self.assertEqual(frame.pair, 0)

    def test_pair_reports_unequal_unpaired_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run"
            root.mkdir()
            tifffile.imwrite(root / "a_CP.tif", np.ones((3, 3), dtype=np.uint16))
            tifffile.imwrite(root / "b_CP.tif", np.full((3, 3), 2, dtype=np.uint16))
            tifffile.imwrite(root / "c_CM.tif", np.full((3, 3), 3, dtype=np.uint16))
            peem_router.load_peem(
                file=None,
                server_path=str(root),
                csv=None,
                csv_path=None,
                name="unequal",
                session=session,
            )

            summary = peem_router.pair_peem(
                "unequal", PeemPairRequest(mode="CP_CM"), session=session
            )

            self.assertEqual(summary.n_pairs, 1)
            self.assertEqual(summary.unpaired_count, 1)
            self.assertEqual(
                peem_router.get_meta("unequal", session=session).unpaired_count, 1
            )

    def test_meta_rejects_non_pair_processed_cube(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run"
            root.mkdir()
            tifffile.imwrite(root / "a_CP.tif", np.ones((2, 2), dtype=np.uint16))
            peem_router.load_peem(
                file=None,
                server_path=str(root),
                csv=None,
                csv_path=None,
                name="invalid_processed",
                session=session,
            )
            arbitrary = TensorData(
                value=np.ones((1, 2, 2, 2)),
                axes=[np.arange(1), np.arange(2), np.arange(2), np.arange(2)],
                labels=["scan", "spin", "y", "x"],
                units=["", "", "px", "px"],
                data_type="Arbitrary processed data",
                metadata={},
            )
            session.workspace.write_processed_data("invalid_processed", arbitrary)

            with self.assertRaises(HTTPException) as ctx:
                peem_router.get_meta("invalid_processed", session=session)

            self.assertEqual(ctx.exception.status_code, 422)

    def test_processed_frame_route_rejects_invalid_channel_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run"
            root.mkdir()
            tifffile.imwrite(root / "a_CP.tif", np.ones((2, 2), dtype=np.uint16))
            peem_router.load_peem(
                file=None,
                server_path=str(root),
                csv=None,
                csv_path=None,
                name="invalid_shape",
                session=session,
            )
            malformed = TensorData(
                value=np.ones((1, 3, 2, 2)),
                axes=[np.arange(1), np.arange(3), np.arange(2), np.arange(2)],
                labels=["pair", "channel", "y", "x"],
                units=["", "", "px", "px"],
                data_type="Experimental PEEM (paired)",
                metadata={"channel_tags": ["CP", "CM", "extra"]},
            )
            session.workspace.write_processed_data("invalid_shape", malformed)
            app = create_app()
            app.dependency_overrides[current_session] = lambda: session
            client = TestClient(app)

            response = client.get(
                "/api/peem/invalid_shape/frame/0?node=processed&channel=0"
            )

            self.assertEqual(response.status_code, 422, response.text)

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

    def test_meta_accepts_i0_units_and_blank_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run"
            root.mkdir()
            tifffile.imwrite(root / "a.tif", np.ones((2, 2), dtype=np.uint16))
            (root / "run.csv").write_text("frame,I0\n0,1.5 nA\n1,\n2,2.0\n")

            peem_router.load_peem(
                file=None,
                server_path=str(root),
                csv=None,
                csv_path=None,
                name="i0_units",
                session=session,
            )
            meta = peem_router.get_meta("i0_units", session=session)

            self.assertEqual(meta.I0, [1.5, None, 2.0])

    def test_server_path_rejects_float64_expansion_over_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run"
            root.mkdir()
            tifffile.imwrite(root / "a.tif", np.ones((3, 3), dtype=np.uint16))

            original_limit = peem_router.MAX_PEEM_BYTES
            peem_router.MAX_PEEM_BYTES = 64
            try:
                with self.assertRaises(HTTPException) as ctx:
                    peem_router.load_peem(
                        file=None,
                        server_path=str(root),
                        csv=None,
                        csv_path=None,
                        name="too_large",
                        session=session,
                    )
            finally:
                peem_router.MAX_PEEM_BYTES = original_limit

            self.assertEqual(ctx.exception.status_code, 413)

    def test_float64_bytes_for_shape_avoids_int64_overflow(self):
        # np.prod of these dims wraps int64 to a negative / tiny value;
        # math.prod stays large and correctly trips the size limit.
        huge = (2**32 - 1, 2**32 - 1)
        wrapped = int(np.prod(huge)) * 8
        safe = peem_router._float64_bytes_for_shape(huge)
        self.assertGreater(safe, peem_router.MAX_PEEM_BYTES)
        self.assertNotEqual(safe, wrapped)
        self.assertLessEqual(wrapped, peem_router.MAX_PEEM_BYTES)

    def test_server_folder_with_spaces_gets_safe_fallback_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run with spaces"
            root.mkdir()
            tifffile.imwrite(root / "a.tif", np.ones((2, 2), dtype=np.uint16))

            summary = peem_router.load_peem(
                file=None,
                server_path=str(root),
                csv=None,
                csv_path=None,
                name=None,
                session=session,
            )

            self.assertEqual(summary.name, "run_with_spaces")

    def test_invalid_name_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "empty"
            root.mkdir()

            with self.assertRaises(HTTPException) as ctx:
                peem_router.load_peem(
                    file=None,
                    server_path=str(root),
                    csv=None,
                    csv_path=None,
                    name="bad name",
                    session=session,
                )

            self.assertEqual(ctx.exception.status_code, 422)
            self.assertIn("Name must", ctx.exception.detail)

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

    def test_zip_rejects_float64_expansion_after_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            app = create_app()
            app.dependency_overrides[current_session] = lambda: session
            client = TestClient(app)
            frame = io.BytesIO()
            tifffile.imwrite(frame, np.zeros((20, 20), dtype=np.uint16))
            payload = io.BytesIO()
            with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("nested/a.tif", frame.getvalue())
            archive_bytes = payload.getvalue()
            limit = max(len(archive_bytes), len(frame.getvalue())) + 100
            self.assertLess(limit, 20 * 20 * 8)

            original_limit = peem_router.MAX_PEEM_BYTES
            peem_router.MAX_PEEM_BYTES = limit
            try:
                response = client.post(
                    "/api/peem/load",
                    files={"file": ("run.zip", archive_bytes, "application/zip")},
                )
            finally:
                peem_router.MAX_PEEM_BYTES = original_limit

            self.assertEqual(response.status_code, 413, response.text)

    def test_zip_csv_match_prefers_extracted_directory_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            app = create_app()
            app.dependency_overrides[current_session] = lambda: session
            client = TestClient(app)
            payload = io.BytesIO()
            frame = io.BytesIO()
            tifffile.imwrite(frame, np.ones((2, 2), dtype=np.uint16))
            with zipfile.ZipFile(payload, "w") as archive:
                archive.writestr("beam_run/a.tif", frame.getvalue())
                archive.writestr("beam_run/beam_run.csv", "I0\n2.5\n")
                archive.writestr("beam_run/upload_name.csv", "I0\n9.9\n")

            response = client.post(
                "/api/peem/load",
                files={
                    "file": (
                        "upload_name.zip",
                        payload.getvalue(),
                        "application/zip",
                    )
                },
                data={"name": "zip_csv"},
            )

            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["csv_attached"])
            meta = client.get("/api/peem/zip_csv/meta")
            self.assertEqual(meta.status_code, 200, meta.text)
            self.assertEqual(meta.json()["I0"], 2.5)

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
