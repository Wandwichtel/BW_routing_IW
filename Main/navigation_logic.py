from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import subprocess
import math
from pathlib import Path
from typing import Optional, Tuple

import folium
import geopandas as gpd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium
from pyproj import CRS, Transformer
import shapely.geometry

SCRIPT_DIR = Path(__file__).resolve().parent
NAV_FILE = SCRIPT_DIR / "BW_navigation.py"
if not NAV_FILE.exists():
    raise FileNotFoundError(f"Router script not found: {NAV_FILE}")

# Bundle layout support (portable deployments).
# When running from a bundle, we expect:
#   <root>\app\BW_navigation_ui_V2.py  (this file)
#   <root>\data\...                   (optional)
#   <root>\cache\...                  (optional)
BUNDLE_ROOT = SCRIPT_DIR.parent
BUNDLE_DATA_DIR = BUNDLE_ROOT / "data"


def _env_or_local_path(env_key: str, local_path: Path, fallback_abs: str) -> str:
    v = os.environ.get(env_key)
    if v:
        return str(v)
    try:
        if local_path.exists():
            return str(local_path)
    except Exception:
        pass
    return str(fallback_abs)

# Project defaults (BW). Can be overridden via env vars for portable bundles.
DEFAULT_ROADS_GPKG = _env_or_local_path(
    "BW_ROADS_GPKG",
    BUNDLE_DATA_DIR / "roads.gpkg",
    r"C:\Users\jvvj4lj\WORK\ArcGIS_Projects\BW\OSM_roads_IW\OSM_roads_IW_update.gpkg",
)
DEFAULT_DEM_TIF = _env_or_local_path(
    "BW_DEM_TIF",
    BUNDLE_DATA_DIR / "dem.tif",
    r"C:\Users\jvvj4lj\WORK\ArcGIS_Projects\BW\DEM_mosaic.tif",
)
DEFAULT_SLOPE_TIF = _env_or_local_path(
    "BW_SLOPE_TIF",
    BUNDLE_DATA_DIR / "slope.tif",
    r"C:\Users\jvvj4lj\WORK\ArcGIS_Projects\BW\DEM_Slope.tif",
)
DEFAULT_TREES_GPKG = _env_or_local_path(
    "BW_TREES_GPKG",
    BUNDLE_DATA_DIR / "trees.gpkg",
    r"C:\Users\jvvj4lj\WORK\ArcGIS_Projects\BW_DG\trees_IW.gpkg",
)

spec = importlib.util.spec_from_file_location("BW_navigation", NAV_FILE)
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to create import spec for: {NAV_FILE}")
nav = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = nav
spec.loader.exec_module(nav)


LatLon = Tuple[float, float]


def _clear_last_run() -> None:
    # Clear cached outputs so the UI cannot accidentally show routes for an old A/B.
    for k in [
        "last_out_gpkg",
        "last_result",
        "last_start_latlon",
        "last_patient_latlon",
        "last_route_gdf",
        "last_park_gdf",
        "last_debug_parks_gdf",
        "last_debug_routes_gdf",
		"last_custom_alt_route_gdf",
		"last_custom_alt_park_gdf",
		"last_forced_road_route_gdf",
		"last_forced_road_park_gdf",
		"last_tobler_direct_walk_gdf",
		"last_tobler_abseil_segments_latlon",
		"last_tobler_abseil_note_latlon",
    ]:
        st.session_state[k] = None


def _safe_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _cache_signature(cache_dir: Path) -> Tuple[float, ...]:
    # Invalidate when core cache files change.
    cache = nav.CachePaths(cache_dir=cache_dir)
    if cache.meta_json.exists():
        meta_path = cache.meta_json
        graph_path = cache.road_graph_pkl
        nodes_path = cache.road_nodes_npy
    else:
        meta_path = cache.legacy_meta_json
        graph_path = cache.legacy_road_graph_pkl
        nodes_path = cache.legacy_road_nodes_npy
    return (
        _safe_mtime(meta_path),
        _safe_mtime(graph_path),
        _safe_mtime(nodes_path),
        _safe_mtime(cache.walk_graph_pkl),
        _safe_mtime(cache.walk_nodes_npy),
        _safe_mtime(cache.walk_nodes_idx_pkl),
        _safe_mtime(cache.drive_spt_meta_json),
        _safe_mtime(cache.drive_spt_dist_pkl),
        _safe_mtime(cache.drive_spt_prev_pkl),
    )


@st.cache_data(show_spinner=False)
def _heli_lz_centers_from_precomputed_mask_wgs84(
    *,
    cache_dir_str: str,
    patient_lat: float,
    patient_lon: float,
    dem_crs_wkt: str,
    radius_m: float,
    dem_res_m: float,
    lz_radius_m: float,
    flat_lz_radius_m: float,
    tree_clearance_m: float,
    max_slope_deg: float,
    max_relief_m: float,
    max_sites: int,
    slope_src_tag: Optional[str] = None,
) -> list[dict]:
    """Fast-path: read precomputed heli-LZ mask and return nearest candidate centers.

    Returns list of dicts: {lat, lon}.
    If the precomputed mask is missing, returns empty (no slow fallback).
    """
    cache_dir = Path(cache_dir_str)
    patient_xy = nav._wgs84_latlon_to_xy_in_crs(
        float(patient_lat),
        float(patient_lon),
        target_crs_wkt=str(dem_crs_wkt),
    )

    try:
        mask_path = nav.heli_lz_mask_cache_path(
            cache_dir,
            res_m=float(dem_res_m),
            lz_radius_m=float(lz_radius_m),
            flat_lz_radius_m=float(flat_lz_radius_m),
            tree_clearance_m=float(tree_clearance_m),
            max_slope_deg=float(max_slope_deg),
            max_relief_m=float(max_relief_m),
            slope_src_tag=slope_src_tag,
        )
    except Exception:
        return []

    if not Path(mask_path).exists():
        return []

    try:
        mask_win, tr, crs, _res_m_win = nav._read_raster_window_native(
            Path(mask_path),
            center_xy=patient_xy,
            radius_m=float(radius_m),
        )
    except Exception:
        return []

    m = np.asarray(mask_win, dtype=np.float32)
    if m.ndim != 2:
        return []
    ok = np.isfinite(m) & (m >= 0.5)
    rr, cc = np.nonzero(ok)
    if rr.size == 0:
        return []

    # Vectorized rc->xy using affine transform.
    r = rr.astype(np.float64)
    c = cc.astype(np.float64)
    x = float(tr.c) + (c + 0.5) * float(tr.a) + (r + 0.5) * float(tr.b)
    y = float(tr.f) + (c + 0.5) * float(tr.d) + (r + 0.5) * float(tr.e)

    dx = x - float(patient_xy[0])
    dy = y - float(patient_xy[1])
    d2 = dx * dx + dy * dy

    k = int(min(max_sites, int(d2.shape[0])))
    if k <= 0:
        return []

    # Partial selection: k nearest.
    if d2.shape[0] > k:
        sel = np.argpartition(d2, k - 1)[:k]
        sel = sel[np.argsort(d2[sel])]
    else:
        sel = np.argsort(d2)

    # Transform to WGS84.
    try:
        src_crs = CRS.from_wkt(crs.to_wkt()) if crs is not None else CRS.from_wkt(str(dem_crs_wkt))
        tf = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
        lon, lat = tf.transform(x[sel], y[sel])
    except Exception:
        return []

    out: list[dict] = []
    for la, lo in zip(np.asarray(lat, dtype=float).tolist(), np.asarray(lon, dtype=float).tolist()):
        out.append({"lat": float(la), "lon": float(lo)})
    return out


@st.cache_data(show_spinner=False)
def _flat_lz_candidates_dem_window(
    *,
    cache_dir_str: str,
    patient_lat: float,
    patient_lon: float,
    dem_crs_wkt: str,
    radius_m: float,
    dem_res_m: float,
    patch_m: float,
    max_slope_deg: float,
    max_relief_m: float,
    max_sites: int,
    tree_gpkg: str,
    lz_radius_m: float = 10.0,
    tree_clearance_m: float = 10.0,
    max_cells: int = 2_000_000,
) -> Optional[gpd.GeoDataFrame]:
    """Heli landing-zone candidates as r=10m circles in EPSG:4326.

    Constraints:
    - Circle area (approximated via a square filter over the DEM window) must be flat.
    - No tree may be within (lz_radius_m + tree_clearance_m) of the circle center.
    """
    try:
        from scipy import ndimage
    except Exception:
        return None

    try:
        from scipy.spatial import cKDTree
    except Exception:
        cKDTree = None

    cache_dir = Path(cache_dir_str)
    patient_xy = nav._wgs84_latlon_to_xy_in_crs(
        float(patient_lat),
        float(patient_lon),
        target_crs_wkt=str(dem_crs_wkt),
    )

    dem_path = nav.dem_cache_path(cache_dir, res_m=float(dem_res_m))
    if not dem_path.exists():
        return None

    # Fast path: if a whole-area precomputed mask exists, just read a window of it.
    try:
        mask_path = nav.heli_lz_mask_cache_path(
            cache_dir,
            res_m=float(dem_res_m),
            lz_radius_m=float(lz_radius_m),
            tree_clearance_m=float(tree_clearance_m),
            max_slope_deg=float(max_slope_deg),
            max_relief_m=float(max_relief_m),
            slope_src_tag=Path(DEFAULT_SLOPE_TIF).stem if Path(DEFAULT_SLOPE_TIF).exists() else None,
        )
    except Exception:
        mask_path = None

    if mask_path is not None and Path(mask_path).exists():
        try:
            mask_win, tr, crs, res_m_win = nav._read_raster_window_native(
                Path(mask_path),
                center_xy=patient_xy,
                radius_m=float(radius_m),
            )
            m = np.asarray(mask_win, dtype=np.float32)
            ok = np.isfinite(m) & (m >= 0.5)
            if ok.ndim != 2:
                return None
            rc = np.argwhere(ok)
            if rc.size == 0:
                return gpd.GeoDataFrame({"kind": []}, geometry=[], crs="EPSG:4326")

            pts_xy = np.array([nav._rc_to_xy(tr, int(r), int(c)) for (r, c) in rc.tolist()], dtype=np.float64)
            dx = pts_xy[:, 0] - float(patient_xy[0])
            dy = pts_xy[:, 1] - float(patient_xy[1])
            order = np.argsort(dx * dx + dy * dy)
            if len(order) > int(max_sites):
                order = order[: int(max_sites)]

            # Optional: compute min tree distance for tooltip (still cheap).
            tree_xy = None
            try:
                trees_path = Path(tree_gpkg)
                if trees_path.exists():
                    try:
                        trees = gpd.read_file(trees_path, layer="trees")
                    except Exception:
                        trees = gpd.read_file(trees_path)
                    if trees is not None and not trees.empty and trees.geometry is not None:
                        if trees.crs is not None and crs is not None and str(trees.crs) != str(crs):
                            trees = trees.to_crs(crs)
                        trees = trees[~trees.geometry.isna()].copy()
                        trees = trees[~trees.geometry.is_empty].copy()
                        if not trees.empty:
                            tree_xy = np.array([(float(p.x), float(p.y)) for p in trees.geometry], dtype=np.float64)
            except Exception:
                tree_xy = None

            tree_tree = None
            if tree_xy is not None and cKDTree is not None and len(tree_xy) > 0:
                try:
                    tree_tree = cKDTree(tree_xy)
                except Exception:
                    tree_tree = None

            polys = []
            min_tree_dist = []
            for idx in order.tolist():
                x, y = float(pts_xy[int(idx)][0]), float(pts_xy[int(idx)][1])
                polys.append(shapely.geometry.Point(x, y).buffer(float(lz_radius_m), resolution=16))
                if tree_tree is not None:
                    try:
                        d, _ = tree_tree.query(np.array([x, y], dtype=np.float64), k=1)
                        min_tree_dist.append(float(d))
                    except Exception:
                        min_tree_dist.append(float("nan"))
                else:
                    min_tree_dist.append(float("nan"))

            gdf = gpd.GeoDataFrame(
                {
                    "kind": ["lz_circle_r10"] * len(polys),
                    "max_slope_deg": [float("nan")] * len(polys),
                    "relief_m": [float("nan")] * len(polys),
                    "min_tree_dist_m": min_tree_dist,
                },
                geometry=polys,
                crs=str(crs.to_wkt() if crs is not None else dem_crs_wkt),
            )
            return gdf.to_crs("EPSG:4326")
        except Exception:
            # Fall back to slow path.
            pass

    dem_win, tr, crs, res_m_win = nav._read_raster_window_native(
        dem_path,
        center_xy=patient_xy,
        radius_m=float(radius_m),
    )
    dem = np.asarray(dem_win, dtype=np.float32)
    if dem.ndim != 2:
        return None
    rows, cols = int(dem.shape[0]), int(dem.shape[1])
    if rows * cols > int(max_cells):
        return None

    res_m_win = float(res_m_win)
    if res_m_win <= 0:
        return None

    # Window size (cells) for ~circle diameter, enforce odd.
    diameter_m = float(lz_radius_m) * 2.0
    win = int(max(1, int(math.ceil(float(diameter_m) / float(res_m_win)))))
    if win % 2 == 0:
        win += 1
    margin = win // 2

    dem_for_max = dem.copy()
    dem_for_min = dem.copy()
    dem_for_max[~np.isfinite(dem_for_max)] = -np.inf
    dem_for_min[~np.isfinite(dem_for_min)] = np.inf
    local_max = ndimage.maximum_filter(dem_for_max, size=win, mode="nearest")
    local_min = ndimage.minimum_filter(dem_for_min, size=win, mode="nearest")
    relief = local_max - local_min

    with np.errstate(invalid="ignore", divide="ignore"):
        dz_dy, dz_dx = np.gradient(dem, res_m_win, res_m_win)
        slope_rise_run = np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)
        slope_deg = np.rad2deg(np.arctan(slope_rise_run)).astype(np.float32)
    slope_deg[~np.isfinite(slope_deg)] = np.inf
    slope_max = ndimage.maximum_filter(slope_deg, size=win, mode="nearest")

    ok = (
        np.isfinite(dem)
        & np.isfinite(relief)
        & (relief <= float(max_relief_m))
        & np.isfinite(slope_max)
        & (slope_max <= float(max_slope_deg))
    )
    if margin > 0:
        ok[:margin, :] = False
        ok[-margin:, :] = False
        ok[:, :margin] = False
        ok[:, -margin:] = False

    rc = np.argwhere(ok)
    if rc.size == 0:
        return gpd.GeoDataFrame({"kind": []}, geometry=[], crs="EPSG:4326")

    pts_xy = np.array([nav._rc_to_xy(tr, int(r), int(c)) for (r, c) in rc.tolist()], dtype=np.float64)
    dx = pts_xy[:, 0] - float(patient_xy[0])
    dy = pts_xy[:, 1] - float(patient_xy[1])
    order = np.argsort(dx * dx + dy * dy)
    if len(order) > int(max_sites):
        order = order[: int(max_sites)]

    # Load trees and build KDTree in the DEM window CRS (if available).
    tree_xy = None
    try:
        trees_path = Path(tree_gpkg)
        if trees_path.exists():
            try:
                trees = gpd.read_file(trees_path, layer="trees")
            except Exception:
                trees = gpd.read_file(trees_path)
            if trees is not None and not trees.empty and trees.geometry is not None:
                if trees.crs is not None and crs is not None and str(trees.crs) != str(crs):
                    trees = trees.to_crs(crs)
                trees = trees[~trees.geometry.isna()].copy()
                trees = trees[~trees.geometry.is_empty].copy()
                if not trees.empty:
                    tree_xy = np.array([(float(p.x), float(p.y)) for p in trees.geometry], dtype=np.float64)
    except Exception:
        tree_xy = None

    tree_tree = None
    if tree_xy is not None and cKDTree is not None and len(tree_xy) > 0:
        try:
            tree_tree = cKDTree(tree_xy)
        except Exception:
            tree_tree = None

    # Create circle polygons (metric buffer) and apply tree clearance.
    polys = []
    max_s = []
    rel = []
    min_tree_dist = []
    clearance_center_m = float(lz_radius_m) + float(tree_clearance_m)
    for idx in order.tolist():
        r, c = int(rc[int(idx)][0]), int(rc[int(idx)][1])
        x, y = float(pts_xy[int(idx)][0]), float(pts_xy[int(idx)][1])
        if tree_tree is not None:
            try:
                d, _ = tree_tree.query(np.array([x, y], dtype=np.float64), k=1)
                d = float(d)
            except Exception:
                d = float("inf")
            if not np.isfinite(d) or d < clearance_center_m:
                continue
            min_tree_dist.append(float(d))
        else:
            min_tree_dist.append(float("nan"))

        polys.append(shapely.geometry.Point(x, y).buffer(float(lz_radius_m), resolution=16))
        max_s.append(float(slope_max[r, c]))
        rel.append(float(relief[r, c]))

    if not polys:
        return gpd.GeoDataFrame({"kind": []}, geometry=[], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(
        {
            "kind": ["lz_circle_r10"] * len(polys),
            "max_slope_deg": max_s,
            "relief_m": rel,
            "min_tree_dist_m": min_tree_dist,
        },
        geometry=polys,
        crs=str(crs.to_wkt() if crs is not None else dem_crs_wkt),
    )
    try:
        return gdf.to_crs("EPSG:4326")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _get_loaded_cache(
    cache_dir_str: str,
    meta_mtime: float,
    graph_mtime: float,
    nodes_mtime: float,
    walk_mtime: float,
    walk_nodes_mtime: float,
    walk_idx_mtime: float,
    spt_meta_mtime: float,
    spt_dist_mtime: float,
    spt_prev_mtime: float,
):
    # NOTE: mtimes are included purely to invalidate the cache when files change.
    return nav.load_cache_resources(Path(cache_dir_str))


def _parse_latlon(text: str) -> LatLon:
    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected 'lat,lon'.")
    return float(parts[0]), float(parts[1])


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cache", default="", help="Cache directory from precompute")
    parser.add_argument("--out", default="", help="Output GeoPackage path")
    parser.add_argument("--center", default=None, help="Map center 'lat,lon' (WGS84)")
    # UI defaults bias toward speed; you can still change them in the app.
    parser.add_argument("--walk-radius-m", type=float, default=3000.0)
    parser.add_argument("--dem-res-m", type=float, default=5.0)
    parser.add_argument(
        "--connect-road-search-m",
        type=float,
        default=nav.Defaults.connect_road_search_m,
    )
    # Streamlit re-runs scripts often; keep parsing permissive.
    args, _ = parser.parse_known_args()
    if args.center is not None:
        args.center = _parse_latlon(args.center)
    return args


def _default_center() -> LatLon:
    d = nav.Defaults()
    return (d.default_start_wgs84_lat, d.default_start_wgs84_lon)


def _add_line_to_map(m: folium.Map, line: gpd.GeoSeries, *, color: str) -> None:
    geom = line.iloc[0]
    if geom is None or geom.is_empty:
        return
    coords = [(lat, lon) for (lon, lat) in list(geom.coords)]
    folium.PolyLine(coords, color=color, weight=5, opacity=0.9).add_to(m)


def _add_line_to_map_dashed(m: folium.Map, line: gpd.GeoSeries, *, color: str) -> None:
    geom = line.iloc[0]
    if geom is None or geom.is_empty:
        return
    coords = [(lat, lon) for (lon, lat) in list(geom.coords)]
    folium.PolyLine(coords, color=color, weight=5, opacity=0.9, dash_array="6,10").add_to(m)


def _iter_latlon_lines(geom) -> list[list[tuple[float, float]]]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    gtype = getattr(geom, "geom_type", "")
    if gtype == "LineString":
        return [[(lat, lon) for (lon, lat) in list(geom.coords)]]
    if gtype == "MultiLineString":
        out: list[list[tuple[float, float]]] = []
        for part in list(geom.geoms):
            out.extend(_iter_latlon_lines(part))
        return out
    return []


def _render_folium_static(m: folium.Map, *, height: int = 650) -> None:
    # Static HTML render avoids Streamlit callbacks that can trigger reruns.
    html = m.get_root().render()
    components.html(html, height=height, scrolling=False)


def _route_metrics_rows(
    gdf_4326: Optional[gpd.GeoDataFrame],
    *,
    crs_wkt_metric: str,
    label: str,
    option_col: str = "option",
    rank_col: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> list[dict]:
    if gdf_4326 is None or getattr(gdf_4326, "empty", True):
        return []
    gdf = gdf_4326
    try:
        gdf_m = gdf.to_crs(crs_wkt_metric)
    except Exception:
        return []
    try:
        dist_m = gdf_m.geometry.length
    except Exception:
        dist_m = np.zeros(len(gdf_m), dtype=float)
    gdf_m = gdf_m.copy()
    gdf_m["_dist_m"] = np.asarray(dist_m, dtype=float)
    if "time_s" not in gdf_m.columns:
        gdf_m["time_s"] = np.nan
    if "mode" not in gdf_m.columns:
        gdf_m["mode"] = ""

    # Back-compat: some exported layers (e.g., debug alternatives) may not have
    # per-row time_s, but do have drive_time_s/walk_time_s/total_time_s.
    need_fill = False
    try:
        ts = np.asarray(gdf_m["time_s"], dtype=float)
        need_fill = not bool(np.any(np.isfinite(ts)))
    except Exception:
        need_fill = True
    if need_fill:
        try:
            mode_s = gdf_m["mode"].astype(str)
        except Exception:
            mode_s = np.asarray([""] * len(gdf_m), dtype=object)

        # Prefer explicit drive/walk times when present.
        if "drive_time_s" in gdf_m.columns and "walk_time_s" in gdf_m.columns:
            try:
                m_drive = mode_s == "drive"
                m_walk = mode_s == "walk"
                gdf_m.loc[m_drive, "time_s"] = gdf_m.loc[m_drive, "drive_time_s"]
                gdf_m.loc[m_walk, "time_s"] = gdf_m.loc[m_walk, "walk_time_s"]
            except Exception:
                pass
        elif "total_time_s" in gdf_m.columns and "drive_time_s" in gdf_m.columns:
            # Fallback: infer walk time as total-drive.
            try:
                m_drive = mode_s == "drive"
                m_walk = mode_s == "walk"
                gdf_m.loc[m_drive, "time_s"] = gdf_m.loc[m_drive, "drive_time_s"]
                gdf_m.loc[m_walk, "time_s"] = gdf_m.loc[m_walk, "total_time_s"] - gdf_m.loc[m_walk, "drive_time_s"]
            except Exception:
                pass

    # Build grouping keys.
    if option_col not in gdf_m.columns:
        gdf_m[option_col] = label
    keys = [option_col]
    if rank_col is not None and rank_col in gdf_m.columns:
        keys.append(rank_col)

    out: list[dict] = []
    # Iterate groups; keep stable ordering.
    grouped = gdf_m.groupby(keys, dropna=False)
    for k, sub in grouped:
        if not isinstance(k, tuple):
            k = (k,)
        opt_val = str(k[0])
        rank_val = None
        if rank_col is not None and len(k) >= 2:
            rank_val = k[1]

        def _sum_for_mode(mode: str) -> tuple[float, float]:
            ss = sub[sub["mode"] == mode]
            t = float(np.nansum(np.asarray(ss.get("time_s"), dtype=float))) if len(ss) else 0.0
            d = float(np.nansum(np.asarray(ss.get("_dist_m"), dtype=float))) if len(ss) else 0.0
            return t, d

        drive_t, drive_d = _sum_for_mode("drive")
        walk_t, walk_d = _sum_for_mode("walk")

        row = {
            "route": str(label),
            "option": opt_val,
            "drive_time_min": drive_t / 60.0,
            "drive_dist_km": drive_d / 1000.0,
            "walk_time_min": walk_t / 60.0,
            "walk_dist_km": walk_d / 1000.0,
            "total_time_min": (drive_t + walk_t) / 60.0,
            "total_dist_km": (drive_d + walk_d) / 1000.0,
        }
        if rank_col is not None:
            row["rank"] = rank_val
        out.append(row)

    # Optional trimming.
    if max_rows is not None and len(out) > int(max_rows):
        out = out[: int(max_rows)]
    return out


# Distinct, high-contrast palette for alternative route overlays.
# (Used for debug alternatives and to distinguish Option 2 from Option 1.)
ALT_ROUTE_COLORS = [
    "#9467bd",  # purple
    "#2ca02c",  # green
    "#d62728",  # red
    "#8c564b",  # brown
    "#e377c2",  # pink
]


def _add_slope_overlay(
    m: folium.Map,
    *,
    slope_deg: np.ndarray,
    transform,
    crs_wkt: str,
    max_deg: float = 60.0,
    target_max_px: int = 900,
    opacity: float = 0.55,
) -> None:
    # Compute the *original* window bounds in source CRS.
    # (We may downsample the image for display, but the bounds must remain unchanged.)
    arr0 = np.asarray(slope_deg, dtype=np.float32)
    h0, w0 = arr0.shape

    left0 = float(transform.c)
    top0 = float(transform.f)
    res_x0 = float(transform.a)
    res_y0 = float(transform.e)  # typically negative
    right0 = left0 + res_x0 * float(w0)
    bottom0 = top0 + res_y0 * float(h0)

    # Downsample for display so we don't push a huge image to the browser.
    arr = arr0
    h, w = arr.shape
    stride = max(1, int(max(h, w) / float(target_max_px)))
    if stride > 1:
        arr = arr[::stride, ::stride]
        h, w = arr.shape

    arr = np.clip(arr, 0.0, float(max_deg))
    ok = np.isfinite(arr)
    t = np.zeros_like(arr, dtype=np.float32)
    t[ok] = arr[ok] / float(max_deg)

    # Color ramp: green (0) -> yellow (0.5) -> red (1)
    # Piecewise linear interpolation.
    r = np.zeros_like(t, dtype=np.float32)
    g = np.zeros_like(t, dtype=np.float32)
    b = np.zeros_like(t, dtype=np.float32)

    lo = t <= 0.5
    hi = ~lo

    # 0..0.5: green -> yellow (increase red)
    r[lo] = (t[lo] / 0.5)
    g[lo] = 1.0
    b[lo] = 0.0

    # 0.5..1: yellow -> red (decrease green)
    r[hi] = 1.0
    g[hi] = 1.0 - ((t[hi] - 0.5) / 0.5)
    b[hi] = 0.0

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(r * 255.0, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(g * 255.0, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(b * 255.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(ok, 255, 0).astype(np.uint8)

    # Compute bounds in WGS84 from all four corners to ensure correct ordering.
    src_crs = CRS.from_wkt(crs_wkt)
    tf = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    corners_xy = [
        (left0, top0),
        (right0, top0),
        (right0, bottom0),
        (left0, bottom0),
    ]
    corners_lonlat = [tf.transform(x, y) for (x, y) in corners_xy]
    lons = [c[0] for c in corners_lonlat]
    lats = [c[1] for c in corners_lonlat]
    west, east = float(min(lons)), float(max(lons))
    south, north = float(min(lats)), float(max(lats))

    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=[[south, west], [north, east]],
        opacity=float(opacity),
        name=f"Slope (deg, 0–{float(max_deg):.0f})",
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(m)


def main() -> int:
    st.set_page_config(page_title="BW Navigation", layout="wide")

    args = _get_args()

    col_cfg, col_run = st.columns([2, 3])
    with col_cfg:
        st.subheader("Config")
        default_cache = str(Path(args.cache)) if str(args.cache or "").strip() else str(SCRIPT_DIR / "cache")
        default_out = str(Path(args.out)) if str(args.out or "").strip() else str(SCRIPT_DIR / "bw_route.gpkg")
        cache_text = st.text_input("Cache directory", value=default_cache)
        out_text = st.text_input("Output GeoPackage", value=default_out)
        st.caption("Tip: run precompute once, then reuse the same cache.")

        with st.expander("Precompute (build cache)", expanded=False):
            st.write("Build the road graph cache from your roads GeoPackage + DEM.")
            auto_precompute = st.checkbox(
                "Auto-precompute if cache is missing",
                value=True,
                help="If the cache directory has no meta/graph files, the app will build them automatically using the values below.",
                key="auto_precompute",
            )
            roads_text = st.text_input(
                "Roads GeoPackage",
                value=DEFAULT_ROADS_GPKG,
                key="roads_text",
            )
            roads_layer = st.text_input("Roads layer (optional)", value="", key="roads_layer")
            dem_text = st.text_input(
                "DEM/DSM GeoTIFF",
                value=DEFAULT_DEM_TIF,
                key="dem_text",
            )
            speed_col = st.text_input("Speed column", value="emergency_speed", key="speed_col")
            fclass_col = st.text_input("Road class column", value="fclass", key="fclass_col")
            node_snap_m = st.number_input(
                "Node snap (m)",
                value=float(nav.Defaults.node_snap_m),
                step=0.05,
                key="node_snap_m",
            )

            st.divider()
            st.write("Optional DEM preprocessing (recommended for speed)")
            build_dem_cache = st.checkbox("Also build cached resampled DEM", value=True, key="build_dem_cache")
            build_resampled_slope = st.checkbox(
                "Also build slope + walk-cost caches at that resolution",
                value=True,
                help="One-time whole-area preprocessing. Speeds up routing by avoiding per-run slope/walk-cost computation.",
                key="build_resampled_slope",
            )
            build_road_walk_mask = st.checkbox(
                "Also build road-walk mask cache (walking prefers roads)",
                value=True,
                help="One-time preprocessing. Allows walking to follow roads/paths when available, with cross-country least-cost as fallback.",
                key="build_road_walk_mask",
            )
            build_offroad_to_roads = st.checkbox(
                "Also build offroad-to-road caches (fast walking)",
                value=True,
                help="One-time whole-area preprocessing. Enables fast per-route walking by looking up offroad time/path to the nearest road.",
                key="build_offroad_to_roads",
            )
            dem_cache_res_m = st.number_input(
                "Cached DEM resolution (m)",
                value=float(args.dem_res_m),
                step=1.0,
                help="Should match the routing 'DEM resample (m)' value for best speed.",
                key="dem_cache_res_m",
            )
            build_slope_cache = st.checkbox(
                "Also build native slope cache (best for full 1m)",
                value=False,
                help="Creates a slope raster once so routing at 1m avoids per-run gradient computation.",
                key="build_slope_cache",
            )

            st.divider()
            st.write("Helicopter landing zones (whole-area precompute)")
            st.caption(
                "Precompute a heli landing-zone mask aligned to the cached DEM so the Result map can display candidates instantly. "
                "Defaults enforce a minimum 25×25 m flat, tree-free landing area."
            )
            st.number_input(
                "Landing area size (m) [square side]",
                value=float(st.session_state.get("heli_lz_area_m", 25.0) or 25.0),
                step=1.0,
                min_value=10.0,
                key="heli_lz_area_m",
            )
            st.number_input(
                "Extra tree clearance beyond footprint edge (m)",
                value=float(st.session_state.get("heli_lz_tree_clearance_m", 0.0) or 0.0),
                step=1.0,
                min_value=0.0,
                key="heli_lz_tree_clearance_m",
            )
            st.number_input(
                "Max slope within footprint (deg)",
                value=float(st.session_state.get("heli_lz_max_slope_deg", 7.0) or 7.0),
                step=0.5,
                min_value=0.5,
                key="heli_lz_max_slope_deg",
            )
            st.number_input(
                "Max relief within footprint (m)",
                value=float(st.session_state.get("heli_lz_max_relief_m", 1.0) or 1.0),
                step=0.1,
                min_value=0.0,
                key="heli_lz_max_relief_m",
            )
            st.text_input(
                "Trees GeoPackage (points)",
                value=str(st.session_state.get("heli_lz_trees_path", DEFAULT_TREES_GPKG) or DEFAULT_TREES_GPKG),
                key="heli_lz_trees_path",
            )
            st.text_input(
                "Trees layer (optional)",
                value=str(st.session_state.get("heli_lz_trees_layer", "trees") or "trees"),
                key="heli_lz_trees_layer",
            )

            if st.button("Precompute heli LZ mask", type="secondary", key="precompute_helilz_btn"):
                if not cache_text:
                    st.error("Please set a cache directory first.")
                else:
                    cache_dir3 = Path(cache_text)
                    if not cache_dir3.exists():
                        st.error("Cache directory does not exist. Run the main precompute first.")
                    else:
                        trees_path3 = Path(str(st.session_state.get("heli_lz_trees_path", "") or ""))
                        if not trees_path3.exists():
                            st.error(f"Trees GeoPackage not found: {trees_path3}")
                        else:
                            area_m = float(st.session_state.get("heli_lz_area_m", 25.0) or 25.0)
                            lz_radius_m = float(area_m) / 2.0
                            with st.spinner("Precomputing heli LZ mask (whole area)..."):
                                try:
                                    out_mask = nav.precompute_heli_lz_mask_for_cached_dem(
                                        cache_dir=cache_dir3,
                                        res_m=float(dem_cache_res_m),
                                        trees_gpkg=trees_path3,
                                        trees_layer=str(st.session_state.get("heli_lz_trees_layer", "") or "").strip() or None,
                                        lz_radius_m=float(lz_radius_m),
                                           flat_lz_radius_m=float(2.5),
                                        tree_clearance_m=float(st.session_state.get("heli_lz_tree_clearance_m", 0.0) or 0.0),
                                        max_slope_deg=float(st.session_state.get("heli_lz_max_slope_deg", 7.0) or 7.0),
                                        max_relief_m=float(st.session_state.get("heli_lz_max_relief_m", 1.0) or 1.0),
                                        slope_src_tif=Path(DEFAULT_SLOPE_TIF) if Path(DEFAULT_SLOPE_TIF).exists() else None,
                                        slope_src_tag=Path(DEFAULT_SLOPE_TIF).stem if Path(DEFAULT_SLOPE_TIF).exists() else None,
                                    )
                                    st.success(f"Heli LZ mask written: {out_mask}")
                                except Exception as e:
                                    st.error(f"Heli LZ precompute failed: {e}")

            pre_btn = st.button("Precompute now", type="secondary")
            if pre_btn:
                if not cache_text:
                    st.error("Please set a cache directory first.")
                elif not roads_text or not Path(roads_text).exists():
                    st.error("Roads GeoPackage path does not exist.")
                elif not dem_text or not Path(dem_text).exists():
                    st.error("DEM/DSM path does not exist.")
                else:
                    with st.spinner("Precomputing road graph cache..."):
                        inputs = nav.Inputs(
                            roads_gpkg=Path(roads_text),
                            roads_layer=(roads_layer.strip() or None),
                            dem_tif=Path(dem_text),
                            speed_col=speed_col.strip() or "emergency_speed",
                            fclass_col=fclass_col.strip() or "fclass",
                        )
                        defaults = nav.Defaults(node_snap_m=float(node_snap_m))
                        nav.precompute(inputs, cache_dir=Path(cache_text), defaults=defaults)
                    if build_dem_cache:
                        with st.spinner("Building cached resampled DEM..."):
                            out_dem = nav.dem_cache_path(Path(cache_text), res_m=float(dem_cache_res_m))
                            nav.precompute_dem_resampled(
                                dem_tif=Path(dem_text),
                                out_tif=out_dem,
                                out_res_m=float(dem_cache_res_m),
                            )
                        msg = f"Cache built. DEM: {out_dem}"
                        if build_resampled_slope:
                            with st.spinner("Building slope + walk-cost caches..."):
                                out_slope_res = nav.precompute_slope_deg_for_cached_dem(
                                    cache_dir=Path(cache_text),
                                    res_m=float(dem_cache_res_m),
                                )
                                out_wc = nav.precompute_walk_cost_for_cached_dem(
                                    cache_dir=Path(cache_text),
                                    res_m=float(dem_cache_res_m),
                                )
                                msg += f" | Slope: {out_slope_res.name} | WalkCost: {out_wc.name}"
                            if build_road_walk_mask:
                                with st.spinner("Building road-walk mask cache..."):
                                    inputs2 = nav.Inputs(
                                        roads_gpkg=Path(roads_text),
                                        roads_layer=(roads_layer.strip() or None),
                                        dem_tif=Path(dem_text),
                                        speed_col=speed_col.strip() or "emergency_speed",
                                        fclass_col=fclass_col.strip() or "fclass",
                                    )
                                    out_rm = nav.precompute_road_walk_mask_for_cached_dem(
                                        inputs=inputs2,
                                        cache_dir=Path(cache_text),
                                        res_m=float(dem_cache_res_m),
                                    )
                                msg += f" | RoadMask: {out_rm.name}"
                            if build_offroad_to_roads:
                                with st.spinner("Building offroad-to-road travel-time caches (safe+risky)..."):
                                    inputs2 = nav.Inputs(
                                        roads_gpkg=Path(roads_text),
                                        roads_layer=(roads_layer.strip() or None),
                                        dem_tif=Path(dem_text),
                                        speed_col=speed_col.strip() or "emergency_speed",
                                        fclass_col=fclass_col.strip() or "fclass",
                                    )
                                    out_safe, out_risky = nav.precompute_offroad_to_road_time_for_cached_dem(
                                        inputs=inputs2,
                                        cache_dir=Path(cache_text),
                                        res_m=float(dem_cache_res_m),
                                        safe_slope_max_deg=float(nav.Defaults().safe_slope_max_deg),
                                    )
                                msg += f" | OffroadSafe: {out_safe.name} | OffroadRisky: {out_risky.name}"
                        if build_slope_cache:
                            with st.spinner("Building native slope cache..."):
                                out_slope = nav.slope_cache_path(Path(cache_text))
                                nav.precompute_slope_deg_native(dem_tif=Path(dem_text), out_tif=out_slope)
                            msg += f" | Slope: {out_slope}"
                        st.success(msg)
                    else:
                        if build_slope_cache:
                            with st.spinner("Building native slope cache..."):
                                out_slope = nav.slope_cache_path(Path(cache_text))
                                nav.precompute_slope_deg_native(dem_tif=Path(dem_text), out_tif=out_slope)
                            st.success(f"Cache + slope built successfully. Slope: {out_slope}")
                        else:
                            st.success("Cache built successfully.")
                    st.rerun()

    cache_dir = Path(cache_text) if cache_text else None
    out_gpkg = Path(out_text) if out_text else None

    st.title("Fastest rescue route (drive + walk)")

    # Load cache metadata to know CRS.
    meta = None
    if cache_dir is None:
        st.warning("Set a cache directory to enable routing.")
        return 0

    try:
        loaded = _get_loaded_cache(str(cache_dir), *_cache_signature(cache_dir))
        meta = loaded.meta
    except Exception as e:
        # If desired and possible, build the cache automatically using the configured paths.
        auto = bool(st.session_state.get("auto_precompute", True))
        roads_text2 = str(st.session_state.get("roads_text", "") or "")
        dem_text2 = str(st.session_state.get("dem_text", "") or "")
        if auto and roads_text2 and dem_text2 and Path(roads_text2).exists() and Path(dem_text2).exists():
            with st.spinner("Cache missing or outdated; auto-precomputing..."):
                try:
                    inputs = nav.Inputs(
                        roads_gpkg=Path(roads_text2),
                        roads_layer=(str(st.session_state.get("roads_layer", "") or "").strip() or None),
                        dem_tif=Path(dem_text2),
                        speed_col=str(st.session_state.get("speed_col", "emergency_speed") or "emergency_speed"),
                        fclass_col=str(st.session_state.get("fclass_col", "fclass") or "fclass"),
                    )
                    defaults_pc = nav.Defaults(node_snap_m=float(st.session_state.get("node_snap_m", nav.Defaults.node_snap_m)))
                    nav.precompute(inputs, cache_dir=cache_dir, defaults=defaults_pc)

                    if bool(st.session_state.get("build_dem_cache", True)):
                        out_dem = nav.dem_cache_path(cache_dir, res_m=float(st.session_state.get("dem_cache_res_m", args.dem_res_m)))
                        nav.precompute_dem_resampled(
                            dem_tif=Path(dem_text2),
                            out_tif=out_dem,
                            out_res_m=float(st.session_state.get("dem_cache_res_m", args.dem_res_m)),
                        )
                        if bool(st.session_state.get("build_resampled_slope", True)):
                            nav.precompute_slope_deg_for_cached_dem(
                                cache_dir=cache_dir,
                                res_m=float(st.session_state.get("dem_cache_res_m", args.dem_res_m)),
                            )
                            nav.precompute_walk_cost_for_cached_dem(
                                cache_dir=cache_dir,
                                res_m=float(st.session_state.get("dem_cache_res_m", args.dem_res_m)),
                            )
                            if bool(st.session_state.get("build_road_walk_mask", True)):
                                nav.precompute_road_walk_mask_for_cached_dem(
                                    inputs=inputs,
                                    cache_dir=cache_dir,
                                    res_m=float(st.session_state.get("dem_cache_res_m", args.dem_res_m)),
                                )
                            if bool(st.session_state.get("build_offroad_to_roads", True)):
                                nav.precompute_offroad_to_road_time_for_cached_dem(
                                    inputs=inputs,
                                    cache_dir=cache_dir,
                                    res_m=float(st.session_state.get("dem_cache_res_m", args.dem_res_m)),
                                    safe_slope_max_deg=float(nav.Defaults().safe_slope_max_deg),
                                )
                    if bool(st.session_state.get("build_slope_cache", False)):
                        out_slope = nav.slope_cache_path(cache_dir)
                        nav.precompute_slope_deg_native(dem_tif=Path(dem_text2), out_tif=out_slope)

                    # Reload cache after precompute so subsequent logic uses the fresh graph/meta.
                    loaded = _get_loaded_cache(str(cache_dir), *_cache_signature(cache_dir))
                    meta = loaded.meta
                except Exception as e2:
                    st.error(f"Auto-precompute failed: {e2}")
                    st.code(
                        f"python BW_navigation.py precompute --roads <roads.gpkg> --dem <DEM.tif> --cache {cache_dir}",
                        language="text",
                    )
                    return 0

                # Cache provenance / mismatch warning (prevents using the wrong road network silently).
                try:
                    cfg_roads = str(st.session_state.get("roads_text", DEFAULT_ROADS_GPKG) or DEFAULT_ROADS_GPKG)
                    cfg_dem = str(st.session_state.get("dem_text", DEFAULT_DEM_TIF) or DEFAULT_DEM_TIF)
                    meta_roads = str(getattr(meta, "roads_gpkg", ""))
                    meta_dem = str(getattr(meta, "dem_tif", ""))
                    if meta_roads and Path(cfg_roads).exists() and os.path.normcase(meta_roads) != os.path.normcase(cfg_roads):
                        st.warning(
                            "Cache was built from a different roads file than the UI is configured for. "
                            "Rebuild cache to ensure `emergency_speed=0` closures are respected."
                        )
                        st.write({"cache_roads_gpkg": meta_roads, "ui_roads_gpkg": cfg_roads})
                    if meta_dem and Path(cfg_dem).exists() and os.path.normcase(meta_dem) != os.path.normcase(cfg_dem):
                        st.warning(
                            "Cache was built from a different DEM than the UI is configured for. "
                            "Rebuild cache to avoid CRS/extent mismatches."
                        )
                        st.write({"cache_dem_tif": meta_dem, "ui_dem_tif": cfg_dem})
                    # Show diagnostics from the cached graph.
                    try:
                        seen = int(loaded.graph.graph.get("seen_features", -1))
                        sk0 = int(loaded.graph.graph.get("skipped_speed0", -1))
                        blk = int(loaded.graph.graph.get("blocked_edges", -1))
                        rm = int(loaded.graph.graph.get("removed_blocked_edges", -1))
                        if seen >= 0 or sk0 >= 0:
                            st.caption(
                                f"Graph build diagnostics: seen_features={seen}, skipped_speed0={sk0}, blocked_edges={blk}, removed_blocked_edges={rm}"
                            )
                    except Exception:
                        pass
                except Exception:
                    pass
            st.success("Cache built. Reloading...")
            st.rerun()

        st.error(str(e))
        st.code(
            f"python BW_navigation.py precompute --roads <roads.gpkg> --dem <DEM.tif> --cache {cache_dir}",
            language="text",
        )
        return 0

    if "start" not in st.session_state:
        st.session_state["start"] = None
    if "patient" not in st.session_state:
        st.session_state["patient"] = None

    # Persist the last successful run so results survive Streamlit reruns.
    if "last_out_gpkg" not in st.session_state:
        st.session_state["last_out_gpkg"] = None
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_start_latlon" not in st.session_state:
        st.session_state["last_start_latlon"] = None
    if "last_patient_latlon" not in st.session_state:
        st.session_state["last_patient_latlon"] = None

    # Alternatives are now automatic; legacy debug_top_k removed.
    try:
        st.session_state.pop("debug_top_k", None)
        st.session_state.pop("last_debug_top_k", None)
    except Exception:
        pass

    if "custom_alt_latlon" not in st.session_state:
        st.session_state["custom_alt_latlon"] = None
    if "custom_alt_option" not in st.session_state:
        st.session_state["custom_alt_option"] = "option1"
    if "custom_alt_avoid_walk_overlap" not in st.session_state:
        st.session_state["custom_alt_avoid_walk_overlap"] = False
    if "custom_alt_tobler_line" not in st.session_state:
        st.session_state["custom_alt_tobler_line"] = False

    if "forced_road_latlon" not in st.session_state:
        st.session_state["forced_road_latlon"] = None

    if "show_heli_lz" not in st.session_state:
        st.session_state["show_heli_lz"] = False

    # Helicopter LZ parameters (used for both preprocessing + display).
    if "heli_lz_area_m" not in st.session_state:
        # Default minimum requirement: 25m x 25m flat and without trees.
        st.session_state["heli_lz_area_m"] = 25.0
    if "heli_lz_tree_clearance_m" not in st.session_state:
        # Extra clearance beyond the landing footprint edge.
        st.session_state["heli_lz_tree_clearance_m"] = 0.0
    if "heli_lz_max_slope_deg" not in st.session_state:
        st.session_state["heli_lz_max_slope_deg"] = 7.0
    if "heli_lz_max_relief_m" not in st.session_state:
        st.session_state["heli_lz_max_relief_m"] = 1.0
    if "heli_lz_trees_path" not in st.session_state:
        st.session_state["heli_lz_trees_path"] = str(DEFAULT_TREES_GPKG)
    if "heli_lz_trees_layer" not in st.session_state:
        st.session_state["heli_lz_trees_layer"] = "trees"

    if "last_tobler_direct_walk_gdf" not in st.session_state:
        st.session_state["last_tobler_direct_walk_gdf"] = None
    if "last_tobler_abseil_segments_latlon" not in st.session_state:
        st.session_state["last_tobler_abseil_segments_latlon"] = None
    if "last_tobler_abseil_note_latlon" not in st.session_state:
        st.session_state["last_tobler_abseil_note_latlon"] = None

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.subheader("Points")
        st.write("Click the map once, then assign the last click to A or B.")

        if st.session_state["start"] is None:
            st.info(
                f"Start (A) default: {nav.Defaults().default_start_wgs84_lat}, {nav.Defaults().default_start_wgs84_lon}"
            )
        else:
            start = st.session_state["start"]
            st.success(f"Start (A): {start[0]:.6f}, {start[1]:.6f}")

        if st.session_state["patient"] is None:
            st.warning("Patient (B) not set yet")
        else:
            patient = st.session_state["patient"]
            st.success(f"Patient (B): {patient[0]:.6f}, {patient[1]:.6f}")

        st.subheader("Routing")
        st.caption(
            "Full 1m DEM is possible but expensive: keep walk radius modest (e.g. 1000–2500 m), "
            "and consider building the native slope cache in Precompute."
        )
        walk_radius_m = st.number_input("Walk radius (m)", value=float(args.walk_radius_m), step=250.0)
        dem_res_m = st.number_input("DEM resample (m)", value=float(args.dem_res_m), step=1.0)
        connect_road_search_m = st.number_input(
            "Snap to road max (m)", value=float(args.connect_road_search_m), step=25.0
        )
        show_slope = st.checkbox("Show slope overlay (degrees)", value=False)

        if st.button(
            "Precompute driving from Start (A)",
            help="Build a driving shortest-path tree from the current Start (A). Big speedup when Start stays fixed.",
        ):
            if cache_dir is None:
                st.error("Set a cache directory first.")
                st.stop()

            if st.session_state["start"] is None:
                d = nav.Defaults()
                start_lat, start_lon = d.default_start_wgs84_lat, d.default_start_wgs84_lon
            else:
                start_lat, start_lon = st.session_state["start"]

            start_xy2 = nav._wgs84_latlon_to_xy_in_crs(start_lat, start_lon, target_crs_wkt=meta.dem_crs_wkt)
            with st.spinner("Precomputing drive SPT (can take a bit)..."):
                loaded2 = _get_loaded_cache(str(cache_dir), *_cache_signature(cache_dir))
                nav.precompute_drive_spt(
                    cache_dir=cache_dir,
                    start_xy=start_xy2,
                    defaults=nav.Defaults(connect_road_search_m=float(connect_road_search_m)),
                    cache_resources=loaded2,
                )
            st.success("Drive SPT built. Routes from this start should be faster.")
            st.rerun()

        run_disabled = st.session_state["patient"] is None or out_gpkg is None
        if out_gpkg is None:
            st.warning("Set an output GeoPackage path to run.")
        run_btn = st.button("Run route", type="primary", disabled=run_disabled)

    with col_left:
        st.subheader("Map")
        center = args.center if args.center is not None else _default_center()
        m = folium.Map(location=[center[0], center[1]], zoom_start=13, control_scale=True, tiles=None)

        folium.TileLayer(
            tiles="OpenStreetMap",
            name="OpenStreetMap",
            overlay=False,
            control=True,
            show=True,
        ).add_to(m)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            name="Satellite (Esri)",
            attr="Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
            overlay=False,
            control=True,
            show=False,
            max_zoom=19,
        ).add_to(m)

        # Show current points.
        if st.session_state["start"] is not None:
            start = st.session_state["start"]
            folium.Marker(
                location=[start[0], start[1]],
                tooltip="Start (A)",
                icon=folium.Icon(color="green"),
            ).add_to(m)
        else:
            d = nav.Defaults()
            folium.Marker(
                location=[d.default_start_wgs84_lat, d.default_start_wgs84_lon],
                tooltip="Start default (A)",
                icon=folium.Icon(color="green"),
            ).add_to(m)

        if st.session_state["patient"] is not None:
            patient = st.session_state["patient"]
            folium.Marker(
                location=[patient[0], patient[1]],
                tooltip="Patient (B)",
                icon=folium.Icon(color="red"),
            ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        map_state = st_folium(m, height=650, width=None, key="pick_map")
        last = map_state.get("last_clicked") if isinstance(map_state, dict) else None

        assign_cols = st.columns(2)
        with assign_cols[0]:
            if st.button("Use last click as Start (A)", disabled=last is None):
                st.session_state["start"] = (float(last["lat"]), float(last["lng"]))
                _clear_last_run()
                st.rerun()
        with assign_cols[1]:
            if st.button("Use last click as Patient (B)", disabled=last is None):
                st.session_state["patient"] = (float(last["lat"]), float(last["lng"]))
                _clear_last_run()
                st.rerun()

    if run_btn:
        defaults = nav.Defaults(
            walk_radius_m=float(walk_radius_m),
            dem_res_m=float(dem_res_m),
            connect_road_search_m=float(connect_road_search_m),
        )

        if st.session_state["start"] is None:
            d = nav.Defaults()
            start_lat, start_lon = d.default_start_wgs84_lat, d.default_start_wgs84_lon
        else:
            start_lat, start_lon = st.session_state["start"]

        patient_lat, patient_lon = st.session_state["patient"]

        start_xy = nav._wgs84_latlon_to_xy_in_crs(start_lat, start_lon, target_crs_wkt=meta.dem_crs_wkt)
        patient_xy = nav._wgs84_latlon_to_xy_in_crs(
            patient_lat, patient_lon, target_crs_wkt=meta.dem_crs_wkt
        )

        with st.spinner("Routing..."):
            try:
                # Reuse loaded cache (graph + KD-tree) for speed.
                loaded2 = _get_loaded_cache(str(cache_dir), *_cache_signature(cache_dir))
                result = nav.route(
                    cache_dir=cache_dir,
                    start_xy=start_xy,
                    patient_xy=patient_xy,
                    out_gpkg=out_gpkg,
                    defaults=defaults,
                    cache_resources=loaded2,
                )
            except Exception as e:
                st.error(str(e))
                st.stop()

        # Read layers once and keep them in memory across reruns.
        try:
            route_gdf = gpd.read_file(out_gpkg, layer="route").to_crs("EPSG:4326")
        except Exception:
            route_gdf = None
        try:
            park_gdf = gpd.read_file(out_gpkg, layer="park").to_crs("EPSG:4326")
        except Exception:
            park_gdf = None
        try:
            debug_parks_gdf = gpd.read_file(out_gpkg, layer="debug_parks").to_crs("EPSG:4326")
        except Exception:
            debug_parks_gdf = None
        try:
            debug_routes_gdf = gpd.read_file(out_gpkg, layer="debug_routes").to_crs("EPSG:4326")
        except Exception:
            debug_routes_gdf = None

        st.session_state["last_out_gpkg"] = str(out_gpkg)
        st.session_state["last_result"] = dict(result)
        st.session_state["last_start_latlon"] = (float(start_lat), float(start_lon))
        st.session_state["last_patient_latlon"] = (float(patient_lat), float(patient_lon))
        st.session_state["last_show_slope"] = bool(show_slope)
        st.session_state["last_route_gdf"] = route_gdf
        st.session_state["last_park_gdf"] = park_gdf
        st.session_state["last_debug_parks_gdf"] = debug_parks_gdf
        st.session_state["last_debug_routes_gdf"] = debug_routes_gdf
        # Clear derived/interactive outputs on fresh base runs.
        st.session_state["last_custom_alt_route_gdf"] = None
        st.session_state["last_custom_alt_park_gdf"] = None
        st.session_state["last_forced_road_route_gdf"] = None
        st.session_state["last_forced_road_park_gdf"] = None
        st.session_state["last_tobler_direct_walk_gdf"] = None
        st.session_state["last_tobler_abseil_segments_latlon"] = None
        st.session_state["last_tobler_abseil_note_latlon"] = None


    # Always render the last result (if available), so it doesn't disappear on reruns.
    last_out = st.session_state.get("last_out_gpkg")
    last_result = st.session_state.get("last_result")
    last_start = st.session_state.get("last_start_latlon")
    last_patient = st.session_state.get("last_patient_latlon")
    last_show_slope = bool(st.session_state.get("last_show_slope", False))
    last_route_gdf = st.session_state.get("last_route_gdf")
    last_park_gdf = st.session_state.get("last_park_gdf")
    last_debug_parks_gdf = st.session_state.get("last_debug_parks_gdf")
    last_debug_routes_gdf = st.session_state.get("last_debug_routes_gdf")

    # If the user has selected a new patient but hasn't re-run routing yet, warn clearly.
    try:
        cur_patient = st.session_state.get("patient")
        if cur_patient is not None and last_patient is not None:
            if abs(float(cur_patient[0]) - float(last_patient[0])) > 1e-9 or abs(float(cur_patient[1]) - float(last_patient[1])) > 1e-9:
                st.warning(
                    "Displayed results correspond to the last computed Patient (B), not the currently selected B. "
                    "Click 'Run route' to recompute options and alternatives for the new B."
                )
    except Exception:
        pass

    if last_out and last_result and last_start and last_patient and Path(last_out).exists():
        st.subheader("Result")

        safe_thr_deg = float(last_result.get("safe_slope_max_deg", 35.0) or 35.0)

        if last_result.get("walk_time_model") == "tobler_signed_stepwise":
            st.caption("Walking time model: Tobler (signed slope per step) — downhill is faster than uphill.")
        # Option 1 (primary): always fastest road-first access.
        opt1_label = str(last_result.get("option1_label") or "")
        opt1_is_steep = bool(int(last_result.get("option1_is_steep", 0) or 0))
        opt1_total = last_result.get("option1_total_time_s", last_result.get("risky_total_time_s", last_result.get("total_time_s")))
        opt1_drive = last_result.get("option1_drive_time_s", last_result.get("risky_drive_time_s", last_result.get("drive_time_s")))
        opt1_walk = last_result.get("option1_walk_time_s", last_result.get("risky_walk_time_s", last_result.get("walk_time_s")))
        opt1_maxs = last_result.get("option1_max_slope_deg", last_result.get("risky_max_slope_deg"))
        opt1_asc = last_result.get("option1_walk_ascent_m", last_result.get("risky_walk_ascent_m"))
        opt1_desc = last_result.get("option1_walk_descent_m", last_result.get("risky_walk_descent_m"))

        extra1 = ""
        try:
            if opt1_maxs is not None and np.isfinite(float(opt1_maxs)):
                extra1 = f" (max slope ~{float(opt1_maxs):.1f}°)"
        except Exception:
            pass
        try:
            if opt1_asc is not None and opt1_desc is not None and np.isfinite(float(opt1_asc)) and np.isfinite(float(opt1_desc)):
                extra1 = f"{extra1} (walk +{float(opt1_asc):.0f} m / -{float(opt1_desc):.0f} m)"
        except Exception:
            pass

        if opt1_is_steep:
            st.warning(
                f"Option 1 ({opt1_label}): {float(opt1_total)/60.0:.1f} min "
                f"(drive {float(opt1_drive)/60.0:.1f} / walk {float(opt1_walk)/60.0:.1f})"
                f"{extra1}"
            )
        else:
            st.success(
                f"Option 1 ({opt1_label}): {float(opt1_total)/60.0:.1f} min "
                f"(drive {float(opt1_drive)/60.0:.1f} / walk {float(opt1_walk)/60.0:.1f})"
                f"{extra1}"
            )

        # Option 2 (alternate): opposite-type alternative.
        if "option2_total_time_s" in last_result:
            opt2_label = str(last_result.get("option2_label") or "")
            opt2_is_steep = bool(int(last_result.get("option2_is_steep", 0) or 0))
            opt2_total = float(last_result.get("option2_total_time_s"))
            opt2_drive = float(last_result.get("option2_drive_time_s"))
            opt2_walk = float(last_result.get("option2_walk_time_s"))
            opt2_maxs = last_result.get("option2_max_slope_deg")
            opt2_asc = last_result.get("option2_walk_ascent_m")
            opt2_desc = last_result.get("option2_walk_descent_m")

            extra2 = ""
            try:
                if opt2_maxs is not None and np.isfinite(float(opt2_maxs)):
                    extra2 = f" (max slope ~{float(opt2_maxs):.1f}°)"
            except Exception:
                pass
            try:
                if opt2_asc is not None and opt2_desc is not None and np.isfinite(float(opt2_asc)) and np.isfinite(float(opt2_desc)):
                    extra2 = f"{extra2} (walk +{float(opt2_asc):.0f} m / -{float(opt2_desc):.0f} m)"
            except Exception:
                pass

            if opt2_is_steep:
                st.warning(
                    f"Option 2 ({opt2_label}): {opt2_total/60.0:.1f} min "
                    f"(drive {opt2_drive/60.0:.1f} / walk {opt2_walk/60.0:.1f})"
                    f"{extra2}"
                )
            else:
                st.success(
                    f"Option 2 ({opt2_label}): {opt2_total/60.0:.1f} min "
                    f"(drive {opt2_drive/60.0:.1f} / walk {opt2_walk/60.0:.1f})"
                    f"{extra2}"
                )
        else:
            # Explain why option 2 was omitted.
            reason = last_result.get("option2_omitted_reason")
            if reason is not None and str(reason).strip() != "":
                st.info(str(reason))

        out_path = Path(last_out)

        with st.expander("Output layers (diagnostic)", expanded=False):
            try:
                import fiona

                layers = list(fiona.listlayers(str(out_path)))
                st.write("Layers:", layers)
            except Exception as e:
                st.write(f"Could not list layers: {e}")

            try:
                st.write(
                    {
                        **{
                            k: last_result.get(k)
                            for k in [
                                "debug_routes_n",
                                "debug_parks_n",
                                "debug_candidate_walk_nodes_risky_n",
                                "debug_candidate_drive_reached_risky_n",
                            ]
                            if k in last_result
                        },
                    }
                )
                if "debug_errors" in last_result:
                    st.warning("Alternative generation notes:")
                    st.write(last_result.get("debug_errors"))
            except Exception:
                pass
        route = last_route_gdf if last_route_gdf is not None else gpd.read_file(out_path, layer="route").to_crs("EPSG:4326")
        park = last_park_gdf if last_park_gdf is not None else gpd.read_file(out_path, layer="park").to_crs("EPSG:4326")
        debug_parks = last_debug_parks_gdf
        debug_routes = last_debug_routes_gdf

        # Optional custom alternative layers.
        try:
            custom_alt_route = st.session_state.get("last_custom_alt_route_gdf")
        except Exception:
            custom_alt_route = None
        try:
            custom_alt_park = st.session_state.get("last_custom_alt_park_gdf")
        except Exception:
            custom_alt_park = None
        if custom_alt_route is None:
            try:
                custom_alt_route = gpd.read_file(out_path, layer="custom_alt_route").to_crs("EPSG:4326")
            except Exception:
                custom_alt_route = None
        if custom_alt_park is None:
            try:
                custom_alt_park = gpd.read_file(out_path, layer="custom_alt_park").to_crs("EPSG:4326")
            except Exception:
                custom_alt_park = None

        # Optional forced-road layers.
        try:
            forced_road_route = st.session_state.get("last_forced_road_route_gdf")
        except Exception:
            forced_road_route = None
        try:
            forced_road_park = st.session_state.get("last_forced_road_park_gdf")
        except Exception:
            forced_road_park = None
        if forced_road_route is None:
            try:
                ok_fr = int(last_result.get("forced_road_ok", 0) or 0) if last_result is not None else 0
            except Exception:
                ok_fr = 0
            if ok_fr == 1:
                try:
                    forced_road_route = gpd.read_file(out_path, layer="forced_road_route").to_crs("EPSG:4326")
                except Exception:
                    forced_road_route = None
        if forced_road_park is None:
            try:
                ok_fr = int(last_result.get("forced_road_ok", 0) or 0) if last_result is not None else 0
            except Exception:
                ok_fr = 0
            if ok_fr == 1:
                try:
                    forced_road_park = gpd.read_file(out_path, layer="forced_road_park").to_crs("EPSG:4326")
                except Exception:
                    forced_road_park = None

        # Optional direct Tobler walk line (selected point -> Patient).
        try:
            tobler_direct_walk = st.session_state.get("last_tobler_direct_walk_gdf")
        except Exception:
            tobler_direct_walk = None
        if tobler_direct_walk is None:
            try:
                tobler_direct_walk = gpd.read_file(out_path, layer="tobler_direct_walk").to_crs("EPSG:4326")
            except Exception:
                tobler_direct_walk = None

        have_debug_routes = debug_routes is not None and not debug_routes.empty
        have_debug_parks = debug_parks is not None and not debug_parks.empty
        have_debug = bool(have_debug_routes or have_debug_parks)

        if not have_debug_routes:
            st.caption(
                "Alternative routes: none found (often due to a bottleneck within ~2 km of the patient)."
            )
        elif debug_parks is not None and not debug_parks.empty:
            try:
                st.subheader("Alternative routes")
                sub = debug_parks.copy()
                if "rank" in sub.columns:
                    try:
                        sub = sub.sort_values(["rank"])
                    except Exception:
                        pass
                for _, row in sub.iterrows():
                    rank = int(row.get("rank")) if row.get("rank") is not None else -1
                    drive_s = float(row.get("drive_time_s")) if row.get("drive_time_s") is not None else float("nan")
                    walk_s = float(row.get("walk_time_s")) if row.get("walk_time_s") is not None else float("nan")
                    total_s = float(row.get("total_time_s")) if row.get("total_time_s") is not None else float("nan")
                    base_opt = str(row.get("base_option") or "") if "base_option" in sub.columns else ""

                    total_min = float("nan")
                    if np.isfinite(total_s):
                        total_min = float(total_s) / 60.0
                    elif np.isfinite(drive_s) and np.isfinite(walk_s):
                        total_min = (float(drive_s) + float(walk_s)) / 60.0

                    extra = f" from {base_opt}" if base_opt else ""
                    if np.isfinite(total_min):
                        st.caption(
                            f"Alternative {rank}{extra}: {total_min:.1f} min "
                            f"(drive {drive_s/60.0:.1f} / walk {walk_s/60.0:.1f})"
                        )
            except Exception:
                pass

        center2 = (float(last_patient[0]), float(last_patient[1]))
        m2 = folium.Map(location=[center2[0], center2[1]], zoom_start=13, control_scale=True, tiles=None)

        folium.TileLayer(
            tiles="OpenStreetMap",
            name="OpenStreetMap",
            overlay=False,
            control=True,
            show=True,
        ).add_to(m2)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            name="Satellite (Esri)",
            attr="Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
            overlay=False,
            control=True,
            show=False,
            max_zoom=19,
        ).add_to(m2)

        # Optional slope overlay (uses cached slope raster if present).
        if last_show_slope:
            try:
                cache_dir2 = Path(cache_text) if cache_text else None
                if cache_dir2 is not None:
                    slope_path = nav.slope_cache_path(cache_dir2)
                    if slope_path.exists():
                        patient_xy2 = nav._wgs84_latlon_to_xy_in_crs(
                            float(last_patient[0]),
                            float(last_patient[1]),
                            target_crs_wkt=meta.dem_crs_wkt,
                        )
                        slope_win, slope_transform, slope_crs, _ = nav._read_raster_window_native(
                            slope_path,
                            center_xy=patient_xy2,
                            radius_m=float(walk_radius_m),
                        )
                        _add_slope_overlay(
                            m2,
                            slope_deg=slope_win,
                            transform=slope_transform,
                            crs_wkt=slope_crs.to_wkt(),
                        )
                    else:
                        st.info("Slope overlay requested, but slope cache not found. Run precompute-slope once.")
            except Exception as e:
                st.info(f"Slope overlay unavailable: {e}")

        have_opt2 = bool("option2_total_time_s" in last_result)
        opt2_label_for_map = str(last_result.get("option2_label") or "")
        fg_opt1 = folium.FeatureGroup(name=f"Option 1 ({opt1_label})", show=True)
        # Hide Option 2 by default to avoid confusing overplotting (it can be toggled on).
        fg_opt2 = folium.FeatureGroup(name=f"Option 2 ({opt2_label_for_map})", show=False)

        # Automatic alternatives (rank 3/4) are exported as option='alt'.
        fg_alt = folium.FeatureGroup(name="Alternatives (ranks 3–4)", show=bool(have_debug_routes))

        fg_custom_alt = folium.FeatureGroup(name="Custom alternative", show=bool(custom_alt_route is not None and not getattr(custom_alt_route, "empty", True)))

        fg_forced = folium.FeatureGroup(name="Forced road waypoint", show=bool(forced_road_route is not None and not getattr(forced_road_route, "empty", True)))

        fg_tobler_direct = folium.FeatureGroup(name="Direct Tobler walk", show=bool(tobler_direct_walk is not None and not getattr(tobler_direct_walk, "empty", True)))

        # Helicopter landing zones (DEM-only flatness).
        show_lz = bool(st.session_state.get("show_heli_lz", False))
        area_m_ui = float(st.session_state.get("heli_lz_area_m", 25.0) or 25.0)
        lz_radius_ui = float(area_m_ui) / 2.0
        # New spec: cleared area is ~area_m_ui×area_m_ui, but only a 5×5m patch must be flat.
        flat_area_m_ui = 5.0
        flat_radius_ui = float(flat_area_m_ui) / 2.0
        fg_lz = folium.FeatureGroup(
            name=f"Helicopter LZ (tree-free ~{area_m_ui:.0f}×{area_m_ui:.0f} m, flat ~{flat_area_m_ui:.0f}×{flat_area_m_ui:.0f} m)",
            show=show_lz,
        )
        if show_lz:
            try:
                cache_dir2 = Path(cache_text) if cache_text else None
            except Exception:
                cache_dir2 = None
            if cache_dir2 is not None:
                try:
                    # IMPORTANT: use the cached DEM resolution for precomputed masks.
                    dem_res_lz = float(st.session_state.get("dem_cache_res_m", dem_res_m) or dem_res_m)
                except Exception:
                    dem_res_lz = float(dem_res_m)

                centers = _heli_lz_centers_from_precomputed_mask_wgs84(
                    cache_dir_str=str(cache_dir2),
                    patient_lat=float(last_patient[0]),
                    patient_lon=float(last_patient[1]),
                    dem_crs_wkt=str(meta.dem_crs_wkt),
                    radius_m=float(walk_radius_m),
                    dem_res_m=float(dem_res_lz),
                    lz_radius_m=float(lz_radius_ui),
                    flat_lz_radius_m=float(flat_radius_ui),
                    tree_clearance_m=float(st.session_state.get("heli_lz_tree_clearance_m", 0.0) or 0.0),
                    max_slope_deg=float(st.session_state.get("heli_lz_max_slope_deg", 7.0) or 7.0),
                    max_relief_m=float(st.session_state.get("heli_lz_max_relief_m", 1.0) or 1.0),
                    max_sites=200,
                    slope_src_tag=Path(DEFAULT_SLOPE_TIF).stem if Path(DEFAULT_SLOPE_TIF).exists() else None,
                )
                # Draw as lightweight circles (much faster than GeoJSON polygons).
                for c in centers:
                    try:
                        folium.Circle(
                            location=[float(c["lat"]), float(c["lon"])],
                            radius=float(lz_radius_ui),
                            color="#00ff00",
                            weight=1,
                            fill=True,
                            fill_color="#00ff00",
                            fill_opacity=0.10,
                            tooltip="Helicopter LZ",
                        ).add_to(fg_lz)
                    except Exception:
                        continue

        # Option 1 (solid)
        if "option" in route.columns:
            drive1 = route[(route["option"] == "option1") & (route["mode"] == "drive")].geometry
            walk1 = route[(route["option"] == "option1") & (route["mode"] == "walk")].geometry
        else:
            drive1 = route[route["mode"] == "drive"].geometry
            walk1 = route[route["mode"] == "walk"].geometry

        if len(drive1) == 1:
            _add_line_to_map(fg_opt1, drive1, color="#1f77b4")
        if len(walk1) == 1:
            _add_line_to_map(fg_opt1, walk1, color="#ff7f0e")

        # Option 2 (dashed)
        if have_opt2 and "option" in route.columns:
            drive2 = route[(route["option"] == "option2") & (route["mode"] == "drive")].geometry
            walk2 = route[(route["option"] == "option2") & (route["mode"] == "walk")].geometry
            if len(drive2) == 1:
                _add_line_to_map_dashed(fg_opt2, drive2, color="#2ca02c")
            if len(walk2) == 1:
                _add_line_to_map_dashed(fg_opt2, walk2, color="#9467bd")

        folium.Marker(
            location=[float(last_start[0]), float(last_start[1])],
            tooltip="Start (A)",
            icon=folium.Icon(color="green"),
        ).add_to(m2)
        folium.Marker(
            location=[float(last_patient[0]), float(last_patient[1])],
            tooltip="Patient (B)",
            icon=folium.Icon(color="red"),
        ).add_to(m2)

        if not park.empty:
            if "option" in park.columns:
                p1 = park[park["option"] == "option1"]
                if not p1.empty:
                    p = p1.geometry.iloc[0]
                    if opt1_is_steep:
                        folium.Marker(
                            location=[p.y, p.x],
                            tooltip="Park (Option 1 - steep)",
                            icon=folium.Icon(color="orange", icon="exclamation-triangle", prefix="fa"),
                        ).add_to(fg_opt1)
                    else:
                        folium.CircleMarker(
                            location=[p.y, p.x],
                            radius=8,
                            color="#1f77b4",
                            fill=True,
                            fill_opacity=0.9,
                            tooltip="Park (Option 1)",
                        ).add_to(fg_opt1)

                p2 = park[park["option"] == "option2"]
                if not p2.empty:
                    p = p2.geometry.iloc[0]
                    opt2_is_steep_for_marker = bool(int(last_result.get("option2_is_steep", 0) or 0))
                    if opt2_is_steep_for_marker:
                        folium.Marker(
                            location=[p.y, p.x],
                            tooltip="Park (Option 2 - steep)",
                            icon=folium.Icon(color="orange", icon="exclamation-triangle", prefix="fa"),
                        ).add_to(fg_opt2)
                    else:
                        folium.CircleMarker(
                            location=[p.y, p.x],
                            radius=7,
                            color="#2ca02c",
                            fill=True,
                            fill_opacity=0.8,
                            tooltip="Park (Option 2)",
                        ).add_to(fg_opt2)
            else:
                p = park.geometry.iloc[0]
                folium.Marker(location=[p.y, p.x], tooltip="Park", icon=folium.Icon(color="blue")).add_to(m2)

        # Optional alternative-route overlays (top-K per option).
        if debug_routes is not None and not debug_routes.empty:
            try:
                dbg = debug_routes.copy()
                if "option" in dbg.columns:
                    for _, row in dbg.iterrows():
                        geom = row.geometry
                        if geom is None or geom.is_empty:
                            continue
                        option = str(row.get("option", ""))
                        mode = str(row.get("mode", ""))
                        rank = row.get("rank")
                        label = f"{option} rank {rank} {mode}" if rank is not None else f"{option} {mode}"
                        # Color by rank so alternatives show in other colours.
                        try:
                            r_i = int(rank) if rank is not None else 1
                        except Exception:
                            r_i = 1
                        r_i = max(1, r_i)
                        color = ALT_ROUTE_COLORS[(r_i - 1) % len(ALT_ROUTE_COLORS)]
                        dash = "6,10" if mode == "drive" else "2,8"
                        target_fg = fg_alt
                        for coords in _iter_latlon_lines(geom):
                            # Filter out degenerate "direct line" artifacts for walk alternatives.
                            # These typically come from a failed road-walk reconstruction.
                            if mode == "walk" and isinstance(coords, list) and len(coords) <= 2:
                                continue
                            folium.PolyLine(
                                coords,
                                color=color,
                                weight=3 if mode == "drive" else 4,
                                opacity=0.70,
                                dash_array=dash,
                                tooltip=label,
                            ).add_to(target_fg)
            except Exception:
                pass

        if debug_parks is not None and not debug_parks.empty:
            try:
                st.caption("Debug alternatives table (criterion vs reported time).")
                cols = [
                    c
                    for c in [
                        "option",
                        "rank",
                        "selected",
                        "criterion",
                        "drive_time_s",
                        "walk_time_s_criterion",
                        "total_time_s_criterion",
                        "walk_time_s",
                        "total_time_s",
                        "walk_ascent_m",
                        "walk_descent_m",
                        "max_slope_deg",
                    ]
                    if c in debug_parks.columns
                ]
                st.dataframe(
                    debug_parks[cols].sort_values(["option", "rank"]) if cols else debug_parks,
                    use_container_width=True,
                    hide_index=True,
                )

                for _, row in debug_parks.iterrows():
                    p = row.geometry
                    if p is None or p.is_empty:
                        continue
                    option = str(row.get("option", ""))
                    rank = row.get("rank")
                    selected = int(row.get("selected", 0)) if row.get("selected") is not None else 0
                    tooltip = f"{option} rank {rank}" if rank is not None else f"{option} candidate"
                    radius = 9 if selected == 1 else 6
                    color = "#2ca02c" if selected == 1 else "#9467bd"
                    target_fg = fg_alt
                    folium.CircleMarker(
                        location=[p.y, p.x],
                        radius=radius,
                        color=color,
                        fill=True,
                        fill_opacity=0.85,
                        tooltip=tooltip,
                    ).add_to(target_fg)
            except Exception:
                pass

        # Optional custom alternative overlay.
        if custom_alt_route is not None and not custom_alt_route.empty:
            try:
                subd = custom_alt_route[custom_alt_route.get("mode") == "drive"] if "mode" in custom_alt_route.columns else custom_alt_route
                subw = custom_alt_route[custom_alt_route.get("mode") == "walk"] if "mode" in custom_alt_route.columns else None
                if subd is not None and not subd.empty:
                    for geom in list(subd.geometry):
                        for coords in _iter_latlon_lines(geom):
                            folium.PolyLine(coords, color=ALT_ROUTE_COLORS[0], weight=5, opacity=0.85, dash_array="6,10", tooltip="Custom alternative (drive)").add_to(fg_custom_alt)
                if subw is not None and not subw.empty:
                    for geom in list(subw.geometry):
                        for coords in _iter_latlon_lines(geom):
                            folium.PolyLine(coords, color=ALT_ROUTE_COLORS[1], weight=5, opacity=0.85, dash_array="2,8", tooltip="Custom alternative (walk)").add_to(fg_custom_alt)
            except Exception:
                pass

        # Optional forced-road overlay.
        if forced_road_route is not None and not forced_road_route.empty:
            try:
                subd = forced_road_route[forced_road_route.get("mode") == "drive"] if "mode" in forced_road_route.columns else forced_road_route
                subw = forced_road_route[forced_road_route.get("mode") == "walk"] if "mode" in forced_road_route.columns else None
                if subd is not None and not subd.empty:
                    for geom in list(subd.geometry):
                        for coords in _iter_latlon_lines(geom):
                            folium.PolyLine(coords, color="#17becf", weight=5, opacity=0.85, dash_array="1,0", tooltip="Forced-road (drive)").add_to(fg_forced)
                if subw is not None and not subw.empty:
                    for geom in list(subw.geometry):
                        for coords in _iter_latlon_lines(geom):
                            folium.PolyLine(coords, color="#bcbd22", weight=5, opacity=0.85, dash_array="2,8", tooltip="Forced-road (walk)").add_to(fg_forced)
            except Exception:
                pass

        # Optional direct Tobler walk line overlay.
        if tobler_direct_walk is not None and not getattr(tobler_direct_walk, "empty", True):
            try:
                for geom in list(tobler_direct_walk.geometry):
                    for coords in _iter_latlon_lines(geom):
                        folium.PolyLine(
                            coords,
                            color="#ff0000",
                            weight=6,
                            opacity=0.9,
                            tooltip="Direct Tobler walk",
                        ).add_to(fg_tobler_direct)
            except Exception:
                pass

            # Highlight abseil (rappel) segments if available.
            try:
                segs = st.session_state.get("last_tobler_abseil_segments_latlon")
            except Exception:
                segs = None
            try:
                note = st.session_state.get("last_tobler_abseil_note_latlon")
            except Exception:
                note = None
            try:
                if isinstance(segs, list) and segs:
                    for seg in segs:
                        if not isinstance(seg, list) or len(seg) < 2:
                            continue
                        # Yellow underlay + red overlay, both dashed.
                        folium.PolyLine(seg, color="#ffff00", weight=10, opacity=0.95, dash_array="6,10", tooltip="Abseilstrecke!").add_to(fg_tobler_direct)
                        folium.PolyLine(seg, color="#ff0000", weight=6, opacity=0.95, dash_array="6,10", tooltip="Abseilstrecke!").add_to(fg_tobler_direct)
                    if isinstance(note, tuple) and len(note) == 2:
                        folium.Marker(
                            location=[float(note[0]), float(note[1])],
                            tooltip="Abseilstrecke!",
                        ).add_to(fg_tobler_direct)
            except Exception:
                pass

        if forced_road_park is not None and not forced_road_park.empty:
            try:
                p = forced_road_park.geometry.iloc[0]
                folium.CircleMarker(
                    location=[p.y, p.x],
                    radius=8,
                    color="#17becf",
                    fill=True,
                    fill_opacity=0.9,
                    tooltip="Forced-road park",
                ).add_to(fg_forced)
            except Exception:
                pass

        if custom_alt_park is not None and not custom_alt_park.empty:
            try:
                p = custom_alt_park.geometry.iloc[0]
                folium.CircleMarker(
                    location=[p.y, p.x],
                    radius=8,
                    color=ALT_ROUTE_COLORS[0],
                    fill=True,
                    fill_opacity=0.9,
                    tooltip="Custom alternative park",
                ).add_to(fg_custom_alt)
            except Exception:
                pass

        fg_opt1.add_to(m2)
        if have_opt2:
            fg_opt2.add_to(m2)
        if have_debug_routes:
            fg_alt.add_to(m2)
        if custom_alt_route is not None and not getattr(custom_alt_route, "empty", True):
            fg_custom_alt.add_to(m2)
        if forced_road_route is not None and not getattr(forced_road_route, "empty", True):
            fg_forced.add_to(m2)
        if tobler_direct_walk is not None and not getattr(tobler_direct_walk, "empty", True):
            fg_tobler_direct.add_to(m2)
        if show_lz:
            fg_lz.add_to(m2)
        folium.LayerControl(collapsed=False).add_to(m2)

        # Interactive result map so the user can click to choose a divergence point.
        try:
            out_mtime = float(Path(last_out).stat().st_mtime) if last_out else 0.0
        except Exception:
            out_mtime = 0.0
        map_state2 = st_folium(m2, height=650, width=None, key=f"result_map_{int(out_mtime)}")
        last_click2 = map_state2.get("last_clicked") if isinstance(map_state2, dict) else None

        # Metrics summary under the map.
        try:
            metric_crs_wkt = str(meta.dem_crs_wkt)
        except Exception:
            metric_crs_wkt = ""
        if metric_crs_wkt:
            rows_summary: list[dict] = []
            try:
                rows_summary.extend(_route_metrics_rows(route, crs_wkt_metric=metric_crs_wkt, label="route", option_col="option"))
            except Exception:
                pass
            try:
                if custom_alt_route is not None and not custom_alt_route.empty:
                    rows_summary.extend(
                        _route_metrics_rows(
                            custom_alt_route,
                            crs_wkt_metric=metric_crs_wkt,
                            label="custom_alt",
                            option_col="option",
                        )
                    )
            except Exception:
                pass

            try:
                if forced_road_route is not None and not forced_road_route.empty:
                    rows_summary.extend(
                        _route_metrics_rows(
                            forced_road_route,
                            crs_wkt_metric=metric_crs_wkt,
                            label="forced_road",
                            option_col="option",
                        )
                    )
            except Exception:
                pass

            try:
                if tobler_direct_walk is not None and not getattr(tobler_direct_walk, "empty", True):
                    rows_summary.extend(
                        _route_metrics_rows(
                            tobler_direct_walk,
                            crs_wkt_metric=metric_crs_wkt,
                            label="tobler_direct_walk",
                            option_col="option",
                        )
                    )
            except Exception:
                pass

            # Debug alternatives: show up to 5 ranks per option.
            try:
                if debug_routes is not None and not debug_routes.empty and "rank" in debug_routes.columns:
                    dbg = debug_routes.copy()
                    # Keep only a few ranks to avoid an overwhelming table.
                    dbg["rank"] = dbg["rank"].astype(int, errors="ignore") if hasattr(dbg["rank"], "astype") else dbg["rank"]
                    try:
                        dbg = dbg[dbg["rank"].astype(int) <= 5]
                    except Exception:
                        pass
                    rows_summary.extend(
                        _route_metrics_rows(
                            dbg,
                            crs_wkt_metric=metric_crs_wkt,
                            label="debug_alt",
                            option_col="option" if "option" in dbg.columns else "option",
                            rank_col="rank",
                        )
                    )
            except Exception:
                pass

            if rows_summary:
                st.caption("Drive/walk time + distance (from route geometries)")
                # Stable, compact ordering.
                try:
                    rows_summary = sorted(
                        rows_summary,
                        key=lambda r: (
                            str(r.get("route")),
                            str(r.get("option")),
                            int(r.get("rank")) if r.get("rank") is not None and str(r.get("rank")).strip() != "" else -1,
                        ),
                    )
                except Exception:
                    pass
                st.dataframe(rows_summary, use_container_width=True, hide_index=True)

        st.checkbox(
            "Highlight helicopter landing sites near Patient (B) (flat 25×25 m default, tree-free)",
            value=bool(st.session_state.get("show_heli_lz", False)),
            key="show_heli_lz",
            help=(
                "Uses a whole-area precomputed mask when available (recommended). "
                "Defaults require a 25×25 m flat, tree-free area; adjust thresholds in the Precompute panel."
            ),
        )

        try:
            cache_dir2 = Path(cache_text) if cache_text else None
        except Exception:
            cache_dir2 = None
        if cache_dir2 is not None and meta is not None:
            try:
                area_m_ui2 = float(st.session_state.get("heli_lz_area_m", 25.0) or 25.0)
                lz_radius_ui2 = float(area_m_ui2) / 2.0
                dem_res_lz2 = float(st.session_state.get("dem_cache_res_m", dem_res_m) or dem_res_m)
                mask_path2 = nav.heli_lz_mask_cache_path(
                    cache_dir2,
                    res_m=float(dem_res_lz2),
                    lz_radius_m=float(lz_radius_ui2),
                    flat_lz_radius_m=float(2.5),
                    tree_clearance_m=float(st.session_state.get("heli_lz_tree_clearance_m", 0.0) or 0.0),
                    max_slope_deg=float(st.session_state.get("heli_lz_max_slope_deg", 7.0) or 7.0),
                    max_relief_m=float(st.session_state.get("heli_lz_max_relief_m", 1.0) or 1.0),
                    slope_src_tag=Path(DEFAULT_SLOPE_TIF).stem if Path(DEFAULT_SLOPE_TIF).exists() else None,
                )
                if Path(mask_path2).exists():
                    st.caption(f"Heli LZ mask found (fast): {Path(mask_path2).name}")
                else:
                    st.caption("Heli LZ mask not found (slow/empty). Run 'Precompute heli LZ mask' in the Precompute panel.")
            except Exception:
                pass

        st.subheader("Forced road waypoint")
        st.caption("Click the result map near a road. The alternative will be forced to pass through the nearest drivable road node.")
        cols_fr = st.columns(2)
        with cols_fr[0]:
            if st.button("Use last click as forced-road point", disabled=last_click2 is None):
                st.session_state["forced_road_latlon"] = (float(last_click2["lat"]), float(last_click2["lng"]))
                st.rerun()
        with cols_fr[1]:
            fr = st.session_state.get("forced_road_latlon")
            if fr is None:
                st.info("No forced-road point set yet")
            else:
                st.success(f"Forced-road point: {float(fr[0]):.6f}, {float(fr[1]):.6f}")

        if st.button(
            "Compute forced-road alternative",
            type="primary",
            disabled=st.session_state.get("forced_road_latlon") is None,
            key="compute_forced_road",
        ):
            if cache_dir is None:
                st.error("Set a cache directory first.")
                st.stop()
            fr = st.session_state.get("forced_road_latlon")
            if fr is None:
                st.stop()

            defaults2 = nav.Defaults(
                walk_radius_m=float(walk_radius_m),
                dem_res_m=float(dem_res_m),
                connect_road_search_m=float(connect_road_search_m),
            )

            start_lat, start_lon = float(last_start[0]), float(last_start[1])
            patient_lat, patient_lon = float(last_patient[0]), float(last_patient[1])

            start_xy = nav._wgs84_latlon_to_xy_in_crs(start_lat, start_lon, target_crs_wkt=meta.dem_crs_wkt)
            patient_xy = nav._wgs84_latlon_to_xy_in_crs(patient_lat, patient_lon, target_crs_wkt=meta.dem_crs_wkt)
            forced_xy = nav._wgs84_latlon_to_xy_in_crs(float(fr[0]), float(fr[1]), target_crs_wkt=meta.dem_crs_wkt)

            with st.spinner("Computing forced-road alternative..."):
                loaded2 = _get_loaded_cache(str(cache_dir), *_cache_signature(cache_dir))
                result2 = nav.route(
                    cache_dir=cache_dir,
                    start_xy=start_xy,
                    patient_xy=patient_xy,
                    out_gpkg=out_gpkg,
                    defaults=defaults2,
                    cache_resources=loaded2,
                    forced_road_xy=forced_xy,
                )

            try:
                ok = int(result2.get("forced_road_ok", 0) or 0)
            except Exception:
                ok = 0
            if ok == 1:
                try:
                    tt = result2.get("forced_road_total_time_s")
                    st.success(f"Forced-road alternative computed. Total ~{float(tt)/60.0:.1f} min" if tt is not None else "Forced-road alternative computed.")
                except Exception:
                    st.success("Forced-road alternative computed.")
            else:
                st.error(f"Forced-road alternative failed: {result2.get('forced_road_error')}")

            # Refresh layers and store in session state.
            try:
                route_gdf2 = gpd.read_file(out_gpkg, layer="route").to_crs("EPSG:4326")
            except Exception:
                route_gdf2 = None
            try:
                park_gdf2 = gpd.read_file(out_gpkg, layer="park").to_crs("EPSG:4326")
            except Exception:
                park_gdf2 = None
            try:
                dbg_parks2 = gpd.read_file(out_gpkg, layer="debug_parks").to_crs("EPSG:4326")
            except Exception:
                dbg_parks2 = None
            try:
                dbg_routes2 = gpd.read_file(out_gpkg, layer="debug_routes").to_crs("EPSG:4326")
            except Exception:
                dbg_routes2 = None
            try:
                ca_route2 = gpd.read_file(out_gpkg, layer="custom_alt_route").to_crs("EPSG:4326")
            except Exception:
                ca_route2 = None
            try:
                ca_park2 = gpd.read_file(out_gpkg, layer="custom_alt_park").to_crs("EPSG:4326")
            except Exception:
                ca_park2 = None

            try:
                fr_route2 = gpd.read_file(out_gpkg, layer="forced_road_route").to_crs("EPSG:4326")
            except Exception:
                fr_route2 = None
            try:
                fr_park2 = gpd.read_file(out_gpkg, layer="forced_road_park").to_crs("EPSG:4326")
            except Exception:
                fr_park2 = None
            try:
                fr_route2 = gpd.read_file(out_gpkg, layer="forced_road_route").to_crs("EPSG:4326")
            except Exception:
                fr_route2 = None
            try:
                fr_park2 = gpd.read_file(out_gpkg, layer="forced_road_park").to_crs("EPSG:4326")
            except Exception:
                fr_park2 = None

            st.session_state["last_out_gpkg"] = str(out_gpkg)
            st.session_state["last_result"] = dict(result2)
            st.session_state["last_start_latlon"] = (float(start_lat), float(start_lon))
            st.session_state["last_patient_latlon"] = (float(patient_lat), float(patient_lon))
            st.session_state["last_show_slope"] = bool(show_slope)
            st.session_state["last_route_gdf"] = route_gdf2
            st.session_state["last_park_gdf"] = park_gdf2
            st.session_state["last_debug_parks_gdf"] = dbg_parks2
            st.session_state["last_debug_routes_gdf"] = dbg_routes2
            st.session_state["last_custom_alt_route_gdf"] = ca_route2
            st.session_state["last_custom_alt_park_gdf"] = ca_park2
            st.session_state["last_forced_road_route_gdf"] = fr_route2
            st.session_state["last_forced_road_park_gdf"] = fr_park2
            # Direct-walk is not recomputed here; clear to avoid stale display.
            st.session_state["last_tobler_direct_walk_gdf"] = None
            st.session_state["last_tobler_abseil_segments_latlon"] = None
            st.session_state["last_tobler_abseil_note_latlon"] = None
            st.rerun()

        st.subheader("Custom alternative")
        st.caption("Click the result map, then compute an alternative from that point to the patient.")
        opt_choices = ["option1"] + (["option2"] if have_opt2 else [])
        st.session_state["custom_alt_option"] = st.selectbox(
            "Base track",
            options=opt_choices,
            index=0 if st.session_state.get("custom_alt_option") not in opt_choices else opt_choices.index(st.session_state.get("custom_alt_option")),
        )

        cols_ca = st.columns(2)
        with cols_ca[0]:
            if st.button("Use last click as alternative-from point", disabled=last_click2 is None):
                st.session_state["custom_alt_latlon"] = (float(last_click2["lat"]), float(last_click2["lng"]))
                st.rerun()
        with cols_ca[1]:
            ca = st.session_state.get("custom_alt_latlon")
            if ca is None:
                st.info("No alternative-from point set yet")
            else:
                st.success(f"Alternative-from: {float(ca[0]):.6f}, {float(ca[1]):.6f}")

        st.checkbox(
            "Avoid overlap with selected option walk (road-walk segment)",
            value=bool(st.session_state.get("custom_alt_avoid_walk_overlap", False)),
            key="custom_alt_avoid_walk_overlap",
        )

        st.checkbox(
            "Also compute: Tobler walk line from this point to Patient (B)",
            value=bool(st.session_state.get("custom_alt_tobler_line", False)),
            key="custom_alt_tobler_line",
            help=(
                "Computes a Tobler least-cost walking path directly from the selected point to the patient. "
                "It is drawn on the result map in red and written into the output GeoPackage as layer 'tobler_direct_walk'."
            ),
        )

        if st.button(
            "Compute custom alternative",
            type="primary",
            disabled=st.session_state.get("custom_alt_latlon") is None,
        ):
            if cache_dir is None:
                st.error("Set a cache directory first.")
                st.stop()
            ca = st.session_state.get("custom_alt_latlon")
            if ca is None:
                st.stop()

            # Recompute routes with the custom alternative request.
            defaults2 = nav.Defaults(
                walk_radius_m=float(walk_radius_m),
                dem_res_m=float(dem_res_m),
                connect_road_search_m=float(connect_road_search_m),
            )

            start_lat, start_lon = float(last_start[0]), float(last_start[1])
            patient_lat, patient_lon = float(last_patient[0]), float(last_patient[1])

            start_xy = nav._wgs84_latlon_to_xy_in_crs(start_lat, start_lon, target_crs_wkt=meta.dem_crs_wkt)
            patient_xy = nav._wgs84_latlon_to_xy_in_crs(patient_lat, patient_lon, target_crs_wkt=meta.dem_crs_wkt)
            alt_xy = nav._wgs84_latlon_to_xy_in_crs(float(ca[0]), float(ca[1]), target_crs_wkt=meta.dem_crs_wkt)

            with st.spinner("Computing custom alternative..."):
                loaded2 = _get_loaded_cache(str(cache_dir), *_cache_signature(cache_dir))
                result2 = nav.route(
                    cache_dir=cache_dir,
                    start_xy=start_xy,
                    patient_xy=patient_xy,
                    out_gpkg=out_gpkg,
                    defaults=defaults2,
                    cache_resources=loaded2,
                    custom_alt_from_xy=alt_xy,
                    custom_alt_option=str(st.session_state.get("custom_alt_option") or "option1"),
                    custom_alt_avoid_walk_overlap=bool(st.session_state.get("custom_alt_avoid_walk_overlap", False)),
                )

            tobler_direct_walk_gdf_4326 = None
            if bool(st.session_state.get("custom_alt_tobler_line", False)):
                try:
                    thr_deg = float(getattr(defaults2, "rappel_min_slope_deg", nav.Defaults().rappel_min_slope_deg))
                    rapp_v = float(getattr(defaults2, "rappel_speed_mps", nav.Defaults().rappel_speed_mps))
                    cached_dem = nav.dem_cache_path(cache_dir, res_m=float(dem_res_m))
                    if not cached_dem.exists():
                        raise FileNotFoundError(
                            f"Cached DEM not found for res={float(dem_res_m)}m: {cached_dem}. Run precompute first."
                        )
                    dem_win, dem_tr, _dem_crs, dem_res_m = nav._read_raster_window_native(
                        cached_dem,
                        center_xy=patient_xy,
                        radius_m=float(walk_radius_m),
                    )

                    # Directional least-cost path: Tobler per step + downhill rappel override.
                    time_rappel_s, walk_path_xy, path_rc = nav._least_cost_walk_time_and_path_directional_tobler_rappel(
                        dem_win,
                        transform=dem_tr,
                        res_m=float(dem_res_m),
                        start_xy=(float(alt_xy[0]), float(alt_xy[1])),
                        target_xy=(float(patient_xy[0]), float(patient_xy[1])),
                        rappel_min_slope_deg=float(thr_deg),
                        rappel_speed_mps=float(rapp_v),
                    )
                    # Ensure exact endpoints.
                    if walk_path_xy:
                        walk_path_xy[0] = (float(alt_xy[0]), float(alt_xy[1]))
                        walk_path_xy[-1] = (float(patient_xy[0]), float(patient_xy[1]))

                    # Also compute pure signed Tobler time along the chosen path (diagnostic).
                    try:
                        time_tobler_s, asc_m, desc_m = nav._walk_time_along_path_signed_tobler(
                            np.asarray(dem_win, dtype=np.float32),
                            path_rc,
                            res_m=float(dem_res_m),
                        )
                    except Exception:
                        time_tobler_s, asc_m, desc_m = float("nan"), float("nan"), float("nan")

                    # Identify abseil (rappel) segments for map styling.
                    abseil_idxs: list[int] = []
                    step_ortho = float(dem_res_m)
                    step_diag = float(dem_res_m) * float(np.sqrt(2.0))
                    dem_arr = np.asarray(dem_win, dtype=np.float32)
                    for i, ((r0, c0), (r1, c1)) in enumerate(zip(path_rc[:-1], path_rc[1:])):
                        z0 = float(dem_arr[int(r0), int(c0)])
                        z1 = float(dem_arr[int(r1), int(c1)])
                        if not np.isfinite(z0) or not np.isfinite(z1):
                            continue
                        dr = abs(int(r1) - int(r0))
                        dc = abs(int(c1) - int(c0))
                        dist = step_diag if (dr == 1 and dc == 1) else step_ortho
                        dz = z1 - z0
                        slope_deg = float(np.degrees(np.arctan2(abs(dz), dist)))
                        if dz < 0 and slope_deg >= float(thr_deg):
                            abseil_idxs.append(int(i))

                    # Build lat/lon polylines for each contiguous abseil run.
                    abseil_segments_latlon: list[list[tuple[float, float]]] = []
                    note_latlon: tuple[float, float] | None = None
                    if abseil_idxs and walk_path_xy:
                        # Transformer to WGS84 for folium.
                        tf = Transformer.from_crs(CRS.from_wkt(str(meta.dem_crs_wkt)), CRS.from_epsg(4326), always_xy=True)

                        abseil_set = set(abseil_idxs)
                        runs: list[tuple[int, int]] = []
                        start_i = None
                        prev_i = None
                        for i in abseil_idxs:
                            if start_i is None:
                                start_i = i
                                prev_i = i
                                continue
                            if prev_i is not None and i == prev_i + 1:
                                prev_i = i
                            else:
                                runs.append((int(start_i), int(prev_i)))
                                start_i = i
                                prev_i = i
                        if start_i is not None and prev_i is not None:
                            runs.append((int(start_i), int(prev_i)))

                        for a, b in runs:
                            # steps a..b correspond to vertices a..(b+1)
                            seg_xy = walk_path_xy[int(a) : int(b) + 2]
                            seg_latlon: list[tuple[float, float]] = []
                            for (x, y) in seg_xy:
                                lon, lat = tf.transform(float(x), float(y))
                                seg_latlon.append((float(lat), float(lon)))
                            if len(seg_latlon) >= 2:
                                abseil_segments_latlon.append(seg_latlon)

                        # Place the note near the midpoint of the first abseil run.
                        if abseil_segments_latlon:
                            first = abseil_segments_latlon[0]
                            note_latlon = first[len(first) // 2]

                    walk_line = nav._line_from_xy(walk_path_xy) if walk_path_xy is not None and len(walk_path_xy) >= 2 else shapely.geometry.LineString()
                    tobler_gdf = gpd.GeoDataFrame(
                        {
                            "mode": ["walk"],
                            "option": ["tobler_direct"],
                            # Report rappel-aware time (Tobler + rappel overrides).
                            "time_s": [float(time_rappel_s)],
                            "time_min": [float(time_rappel_s) / 60.0],
                            "abseil": [int(1 if abseil_idxs else 0)],
                            "abseil_steps_n": [int(len(abseil_idxs))],
                            "rappel_min_slope_deg": [float(thr_deg)],
                            "rappel_speed_mps": [float(rapp_v)],
                            "walk_time_s_tobler": [float(time_tobler_s)],
                            "walk_ascent_m": [float(asc_m)],
                            "walk_descent_m": [float(desc_m)],
                        },
                        geometry=[walk_line],
                        crs=str(meta.dem_crs_wkt),
                    )
                    # Write (or overwrite) a dedicated layer in the same GeoPackage.
                    tobler_gdf.to_file(out_gpkg, layer="tobler_direct_walk", driver="GPKG")
                    tobler_direct_walk_gdf_4326 = tobler_gdf.to_crs("EPSG:4326")

                    # Persist abseil styling info for the UI.
                    st.session_state["last_tobler_abseil_segments_latlon"] = abseil_segments_latlon if abseil_segments_latlon else None
                    st.session_state["last_tobler_abseil_note_latlon"] = note_latlon

                    st.success(
                        f"Direct Tobler walk line computed (~{float(time_rappel_s) / 60.0:.1f} min) and written to GeoPackage layer 'tobler_direct_walk'."
                    )
                except Exception as e:
                    st.error(f"Direct Tobler walk line failed: {e}")

            try:
                ok = int(result2.get("custom_alt_ok", 0) or 0)
            except Exception:
                ok = 0
            if ok == 1:
                st.success(
                    f"Custom alternative computed. Total ~{float(result2.get('custom_alt_total_time_s'))/60.0:.1f} min"
                    if result2.get("custom_alt_total_time_s") is not None
                    else "Custom alternative computed."
                )
            else:
                st.error(f"Custom alternative failed: {result2.get('custom_alt_error')}")

            # Refresh layers and store in session state.
            try:
                route_gdf2 = gpd.read_file(out_gpkg, layer="route").to_crs("EPSG:4326")
            except Exception:
                route_gdf2 = None
            try:
                park_gdf2 = gpd.read_file(out_gpkg, layer="park").to_crs("EPSG:4326")
            except Exception:
                park_gdf2 = None
            try:
                dbg_parks2 = gpd.read_file(out_gpkg, layer="debug_parks").to_crs("EPSG:4326")
            except Exception:
                dbg_parks2 = None
            try:
                dbg_routes2 = gpd.read_file(out_gpkg, layer="debug_routes").to_crs("EPSG:4326")
            except Exception:
                dbg_routes2 = None
            try:
                ca_route2 = gpd.read_file(out_gpkg, layer="custom_alt_route").to_crs("EPSG:4326")
            except Exception:
                ca_route2 = None
            try:
                ca_park2 = gpd.read_file(out_gpkg, layer="custom_alt_park").to_crs("EPSG:4326")
            except Exception:
                ca_park2 = None

            st.session_state["last_out_gpkg"] = str(out_gpkg)
            st.session_state["last_result"] = dict(result2)
            st.session_state["last_start_latlon"] = (float(start_lat), float(start_lon))
            st.session_state["last_patient_latlon"] = (float(patient_lat), float(patient_lon))
            st.session_state["last_show_slope"] = bool(show_slope)
            st.session_state["last_route_gdf"] = route_gdf2
            st.session_state["last_park_gdf"] = park_gdf2
            st.session_state["last_debug_parks_gdf"] = dbg_parks2
            st.session_state["last_debug_routes_gdf"] = dbg_routes2
            st.session_state["last_custom_alt_route_gdf"] = ca_route2
            st.session_state["last_custom_alt_park_gdf"] = ca_park2

            st.session_state["last_tobler_direct_walk_gdf"] = tobler_direct_walk_gdf_4326
            if not bool(st.session_state.get("custom_alt_tobler_line", False)):
                st.session_state["last_tobler_abseil_segments_latlon"] = None
                st.session_state["last_tobler_abseil_note_latlon"] = None

            # Do not auto-carry forced-road layers across runs unless recomputed.
            st.session_state["last_forced_road_route_gdf"] = None
            st.session_state["last_forced_road_park_gdf"] = None
            st.rerun()

        st.caption(f"Saved to: {out_path}")

    return 0


if __name__ == "__main__":
    # If started as plain Python (e.g., VS Code debugger), relaunch correctly via Streamlit.
    # This prevents noisy "missing ScriptRunContext" warnings and enables session_state.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
    except Exception:
        ctx = None

    if ctx is None:
        raise SystemExit(
            subprocess.call(["streamlit", "run", str(Path(__file__)), "--", *sys.argv[1:]])
        )

    main()
