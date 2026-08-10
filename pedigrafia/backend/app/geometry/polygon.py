"""Primitivas de polígono em **milímetros**.

Este módulo é deliberadamente livre de OpenCV: é numpy puro e determinístico, e é
espelhado ponto a ponto em ``frontend/src/geom/polygon.ts``. Um teste de paridade
compara os dois com tolerância de 1e-6 mm.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def as_points(pts) -> Array:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"esperado array (N,2), recebido {arr.shape}")
    return arr


def signed_area(pts: Array) -> float:
    """Área com sinal (fórmula do cadarço). Positiva = sentido anti-horário."""
    p = as_points(pts)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def area(pts: Array) -> float:
    return abs(signed_area(pts))


def perimeter(pts: Array, closed: bool = True) -> float:
    p = as_points(pts)
    if len(p) < 2:
        return 0.0
    d = np.roll(p, -1, axis=0) - p if closed else p[1:] - p[:-1]
    return float(np.sum(np.linalg.norm(d, axis=1)))


def centroid(pts: Array) -> Array:
    """Centroide de área do polígono (não a média dos vértices)."""
    p = as_points(pts)
    a = signed_area(p)
    if abs(a) < 1e-12:
        return p.mean(axis=0)
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    cx = float(np.sum((x + xn) * cross) / (6.0 * a))
    cy = float(np.sum((y + yn) * cross) / (6.0 * a))
    return np.array([cx, cy])


def second_moments(pts: Array) -> tuple[float, float, float]:
    """Momentos centrais de área (Ixx, Iyy, Ixy) — fórmula exata para polígonos."""
    p = as_points(pts)
    c = centroid(p)
    q = p - c
    x, y = q[:, 0], q[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = 0.5 * np.sum(cross)
    if abs(a) < 1e-12:
        return 0.0, 0.0, 0.0
    ixx = float(np.sum((y * y + y * yn + yn * yn) * cross) / 12.0)
    iyy = float(np.sum((x * x + x * xn + xn * xn) * cross) / 12.0)
    ixy = float(np.sum((x * yn + 2 * x * y + 2 * xn * yn + xn * y) * cross) / 24.0)
    sign = 1.0 if a > 0 else -1.0
    return ixx * sign, iyy * sign, ixy * sign


def principal_axis(pts: Array) -> Array:
    """Direção do eixo principal de maior inércia longitudinal (vetor unitário)."""
    ixx, iyy, ixy = second_moments(pts)
    # Autovetor dominante da matriz de covariância [[iyy, ixy], [ixy, ixx]]
    cov = np.array([[iyy, ixy], [ixy, ixx]], dtype=np.float64)
    if not np.all(np.isfinite(cov)):
        return np.array([0.0, 1.0])
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 1.0])


def resample_closed(pts: Array, step_mm: float) -> Array:
    """Reamostra um contorno fechado com passo constante em mm (arco uniforme)."""
    p = as_points(pts)
    if len(p) < 3 or step_mm <= 0:
        return p
    closed = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(np.sum(seg))
    if total < step_mm * 3:
        return p
    n = max(8, int(round(total / step_mm)))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, n, endpoint=False)
    idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(seg) - 1)
    seg_len = np.where(seg[idx] > 1e-12, seg[idx], 1.0)
    frac = ((targets - cum[idx]) / seg_len)[:, None]
    return closed[idx] + frac * (closed[idx + 1] - closed[idx])


def smooth_closed(pts: Array, window_mm: float, step_mm: float) -> Array:
    """Média móvel circular com janela definida em **mm de arco**.

    Preserva o perímetro/dimensão dentro da tolerância verificada em teste: a janela é
    pequena em relação ao comprimento do pé e a operação é simétrica (não encolhe
    sistematicamente como uma erosão faria).
    """
    p = as_points(pts)
    if len(p) < 8 or window_mm <= 0 or step_mm <= 0:
        return p
    k = int(round(window_mm / step_mm))
    if k < 3:
        return p
    if k % 2 == 0:
        k += 1
    half = k // 2
    kernel = np.ones(k) / k
    ext = np.vstack([p[-half:], p, p[:half]])
    xs = np.convolve(ext[:, 0], kernel, mode="valid")
    ys = np.convolve(ext[:, 1], kernel, mode="valid")
    return np.stack([xs, ys], axis=1)


def _rdp_mask(p: Array, eps: float) -> Array:
    keep = np.zeros(len(p), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(p) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = p[i], p[j]
        ab = b - a
        nrm = float(np.linalg.norm(ab))
        seg = p[i + 1:j]
        if nrm < 1e-12:
            d = np.linalg.norm(seg - a, axis=1)
        else:
            d = np.abs(np.cross(np.broadcast_to(ab, seg.shape), seg - a)) / nrm
        m = int(np.argmax(d))
        if float(d[m]) > eps:
            k = i + 1 + m
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return keep


def simplify_closed(pts: Array, epsilon_mm: float) -> Array:
    """Ramer–Douglas–Peucker sobre contorno fechado, com erro máximo em mm."""
    p = as_points(pts)
    if len(p) < 5 or epsilon_mm <= 0:
        return p
    # Ancora nos dois pontos mais distantes para evitar dependência do ponto inicial.
    d2 = np.sum((p - p.mean(axis=0)) ** 2, axis=1)
    start = int(np.argmax(d2))
    rolled = np.roll(p, -start, axis=0)
    opened = np.vstack([rolled, rolled[:1]])
    keep = _rdp_mask(opened, epsilon_mm)
    out = opened[keep][:-1] if keep[-1] else opened[keep]
    return out if len(out) >= 4 else p


def max_deviation(reference: Array, simplified: Array) -> float:
    """Maior distância de um ponto de ``reference`` ao polígono ``simplified`` (mm)."""
    r = as_points(reference)
    s = as_points(simplified)
    if len(s) < 2:
        return float("inf")
    a = s
    b = np.roll(s, -1, axis=0)
    ab = b - a
    denom = np.sum(ab * ab, axis=1)
    denom = np.where(denom < 1e-12, 1.0, denom)
    worst = 0.0
    for pt in r:
        t = np.clip(np.sum((pt - a) * ab, axis=1) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        worst = max(worst, float(np.min(np.linalg.norm(proj - pt, axis=1))))
    return worst


def project(pts: Array, origin: Array, u: Array, v: Array) -> Array:
    """Converte pontos do plano para o frame local (u, v)."""
    q = as_points(pts) - np.asarray(origin, dtype=np.float64)
    return np.stack([q @ np.asarray(u, float), q @ np.asarray(v, float)], axis=1)


def unproject(uv: Array, origin: Array, u: Array, v: Array) -> Array:
    uv = as_points(uv)
    return (np.asarray(origin, float)
            + uv[:, 0:1] * np.asarray(u, float)
            + uv[:, 1:2] * np.asarray(v, float))


def crossings_at_u(uv: Array, u_value: float) -> Array:
    """Valores de v onde o polígono (em coords locais) cruza a reta u = u_value."""
    p = as_points(uv)
    a = p
    b = np.roll(p, -1, axis=0)
    ua, ub = a[:, 0], b[:, 0]
    # Aresta cruza se u_value está entre ua e ub (semiaberto evita contagem dupla)
    lo = np.minimum(ua, ub)
    hi = np.maximum(ua, ub)
    mask = (lo <= u_value) & (u_value < hi)
    if not np.any(mask):
        # Caso degenerado: exatamente no extremo superior
        mask = (lo < u_value) & (u_value <= hi)
        if not np.any(mask):
            return np.empty(0)
    ua_m, ub_m = ua[mask], ub[mask]
    va, vb = a[mask, 1], b[mask, 1]
    denom = np.where(np.abs(ub_m - ua_m) < 1e-12, 1.0, ub_m - ua_m)
    t = (u_value - ua_m) / denom
    return va + t * (vb - va)


def width_at_u(uv: Array, u_value: float) -> float:
    """Largura total (span externo) da secção transversal em u = u_value."""
    vs = crossings_at_u(uv, u_value)
    if vs.size < 2:
        return 0.0
    return float(np.max(vs) - np.min(vs))


def width_profile(uv: Array, u_start: float, u_end: float, samples: int) -> tuple[Array, Array]:
    us = np.linspace(u_start, u_end, samples)
    ws = np.array([width_at_u(uv, float(u)) for u in us])
    return us, ws


def clip_half_plane(pts: Array, normal: Array, offset: float) -> Array:
    """Sutherland–Hodgman: mantém o subconjunto onde ``dot(p, normal) <= offset``."""
    p = as_points(pts)
    if len(p) < 3:
        return np.empty((0, 2))
    n = np.asarray(normal, dtype=np.float64)
    out: list[np.ndarray] = []
    for i in range(len(p)):
        cur = p[i]
        nxt = p[(i + 1) % len(p)]
        dc = float(cur @ n) - offset
        dn = float(nxt @ n) - offset
        if dc <= 0:
            out.append(cur)
        if (dc > 0) != (dn > 0):
            denom = dc - dn
            if abs(denom) > 1e-12:
                t = dc / denom
                out.append(cur + t * (nxt - cur))
    return np.array(out) if len(out) >= 3 else np.empty((0, 2))


def clip_band_u(pts_uv: Array, u_lo: float, u_hi: float) -> Array:
    """Recorta o polígono (coords locais) à faixa u ∈ [u_lo, u_hi]."""
    poly = clip_half_plane(pts_uv, np.array([1.0, 0.0]), u_hi)
    if len(poly) < 3:
        return poly
    return clip_half_plane(poly, np.array([-1.0, 0.0]), -u_lo)


def point_line_distance(point: Array, a: Array, b: Array) -> float:
    """Distância perpendicular de ``point`` à reta infinita AB."""
    p = np.asarray(point, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ab = b - a
    n = float(np.linalg.norm(ab))
    if n < 1e-12:
        return float(np.linalg.norm(p - a))
    return float(abs(np.cross(ab, p - a)) / n)


def convex_hull(pts: Array) -> Array:
    """Monotone chain. Devolve os vértices no sentido anti-horário."""
    p = as_points(pts)
    if len(p) < 3:
        return p
    order = np.lexsort((p[:, 1], p[:, 0]))
    s = p[order]

    def build(seq):
        out: list[np.ndarray] = []
        for pt in seq:
            while len(out) >= 2:
                o, a = out[-2], out[-1]
                if np.cross(a - o, pt - o) <= 1e-12:
                    out.pop()
                else:
                    break
            out.append(pt)
        return out

    lower = build(s)
    upper = build(s[::-1])
    return np.array(lower[:-1] + upper[:-1])


def max_caliper(pts: Array) -> tuple[float, Array, Array]:
    """Diâmetro do fecho convexo (distância máxima entre dois pontos) e o par."""
    h = convex_hull(pts)
    if len(h) < 2:
        return 0.0, h[0] if len(h) else np.zeros(2), h[0] if len(h) else np.zeros(2)
    best = 0.0
    pa = pb = h[0]
    n = len(h)
    j = 1
    for i in range(n):
        while True:
            nxt = (j + 1) % n
            if float(np.linalg.norm(h[nxt] - h[i])) > float(np.linalg.norm(h[j] - h[i])):
                j = nxt
            else:
                break
        d = float(np.linalg.norm(h[j] - h[i]))
        if d > best:
            best, pa, pb = d, h[i], h[j]
    return best, pa, pb


def is_simple(pts: Array) -> bool:
    """Verifica ausência de auto-interseções (usado na validação da revisão)."""
    p = as_points(pts)
    n = len(p)
    if n < 4:
        return n >= 3

    def seg_intersect(p1, p2, p3, p4) -> bool:
        d1 = np.cross(p4 - p3, p1 - p3)
        d2 = np.cross(p4 - p3, p2 - p3)
        d3 = np.cross(p2 - p1, p3 - p1)
        d4 = np.cross(p2 - p1, p4 - p1)
        return bool(((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)))

    for i in range(n):
        a1, a2 = p[i], p[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue
            b1, b2 = p[j], p[(j + 1) % n]
            if seg_intersect(a1, a2, b1, b2):
                return False
    return True
