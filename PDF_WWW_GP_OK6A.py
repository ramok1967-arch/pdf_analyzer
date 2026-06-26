import io
import os
import tempfile

import numpy as np
import streamlit as st
import hyperspy.api as hs
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve


_trapezoid = getattr(np, "trapezoid", np.trapz)


def als_baseline(y, lam=1e5, p=0.01, niter=10):
    """Asymmetric Least Squares baseline (Eilers, 2003)."""
    y = np.asarray(y, dtype=float)
    L = len(y)
    if L < 3:
        return y.copy()

    diag_main = np.ones(L)
    D = sparse.diags(
        [diag_main[:-2], -2 * diag_main[:-1], diag_main],
        offsets=[0, 1, 2],
        shape=(L - 2, L),
    )
    D = lam * (D.T @ D)

    w = np.ones(L)
    z = y.copy()
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = (W + D).tocsc()
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1.0 - p) * (y <= z)

    return z


st.set_page_config(page_title="PDF Analyzer", layout="wide")


class PDFWebAnalyzer:
    def __init__(self, raw_data: np.ndarray, q_per_pixel: float):
        self.raw_data = np.asarray(raw_data, dtype=float)
        self.q_per_pixel_file = float(q_per_pixel)
        self.q_per_pixel = float(q_per_pixel)

        if self.raw_data.ndim != 2:
            raise ValueError("Obsługiwane są tylko dane 2D.")

        st.sidebar.header("⚙️ Ustawienia analizy")

        # 1. Wycinanie
        h, w = self.raw_data.shape
        max_c = min(h, w) // 2

        if max_c < 50:
            raise ValueError(
                f"Obraz jest zbyt mały do analizy. Minimalny pół-rozmiar cropa to 50 px, dostępne: {max_c}."
            )

        with st.sidebar.expander("✂️ Wycinanie obrazu", expanded=True):
            self.crop_val = st.slider(
                "Rozmiar wycinka (crop)",
                min_value=50,
                max_value=max_c,
                value=max_c,
            )

        # 1b. Kalibracja q/pixel
        with st.sidebar.expander("📏 Kalibracja q/pixel", expanded=True):
            st.caption(
                "Skala odczytana z pliku może być skorygowana mnożnikiem albo nadpisana ręcznie."
            )

            st.markdown(
                f"**Odczytane z pliku:** {self.q_per_pixel_file:.6f} Å⁻¹/pixel"
            )

            self.use_manual_q = st.checkbox("Ustaw q/pixel ręcznie", value=False)

            if self.use_manual_q:
                self.q_per_pixel = st.number_input(
                    "Ręczne q/pixel (Å⁻¹/pixel)",
                    min_value=1e-6,
                    value=float(self.q_per_pixel_file),
                    format="%.6f",
                )
                self.q_scale_correction = self.q_per_pixel / self.q_per_pixel_file
            else:
                self.q_scale_correction = st.slider(
                    "Mnożnik korekcyjny q/pixel",
                    min_value=0.80,
                    max_value=1.20,
                    value=1.00,
                    step=0.005,
                )
                self.q_per_pixel = self.q_per_pixel_file * self.q_scale_correction

            st.markdown(
                f"**Używane w analizie:** {self.q_per_pixel:.6f} Å⁻¹/pixel"
            )

        y0 = h // 2 - self.crop_val
        y1 = h // 2 + self.crop_val
        x0 = w // 2 - self.crop_val
        x1 = w // 2 + self.crop_val

        self.image = self.raw_data[y0:y1, x0:x1]

        if self.image.size == 0:
            raise ValueError("Wybrany crop jest pusty.")

        # 2. Pozycjonowanie środka
        with st.sidebar.expander("📍 Pozycjonowanie środka", expanded=True):
            st.caption("Ustaw środek ręcznie i dopasuj okrąg do pierścienia.")

            off_x = st.number_input("Przesunięcie X", value=0.0, step=0.5)
            off_y = st.number_input("Przesunięcie Y", value=0.0, step=0.5)

            self.center = [
                float(self.image.shape[1] / 2 + off_x),
                float(self.image.shape[0] / 2 + off_y),
            ]

            self.show_guide_circle = st.checkbox(
                "Pokaż okrąg pomocniczy", value=True
            )
            self.guide_radius = st.slider(
                "Promień okręgu pomocniczego",
                min_value=10,
                max_value=self.crop_val,
                value=min(300, self.crop_val),
            )

        # 3. Analizowany sektor
        with st.sidebar.expander("🍕 Analizowany sektor", expanded=True):
            st.caption(
                "Wybierz kierunek analizy i pół-szerokość dwóch przeciwległych sektorów."
            )

            self.target_angle = st.slider("Kąt bazowy", 0, 360, 135)
            self.sector_half_width = st.slider("Pół-szerokość sektora (°)", 1, 90, 30)

        # 4. Maskowanie artefaktów
        with st.sidebar.expander("📐 Maskowanie artefaktów", expanded=True):
            st.caption("Wytnij wewnętrzne części sektorów, np. obszar przesłony.")

            max_mask_limit = max(1, self.crop_val // 2)

            a1_deg = self.target_angle % 360
            a2_deg = (self.target_angle + 180) % 360

            default_rmin = min(
                200, max_mask_limit // 2 if max_mask_limit < 200 else 200
            )

            self.use_mask_1 = st.checkbox(
                f"Włącz maskę dla sektora 1 ({a1_deg}°)", value=False
            )
            if self.use_mask_1:
                self.r_min_1 = st.slider(
                    "Maska od środka - sektor 1",
                    0,
                    max_mask_limit,
                    default_rmin,
                )
            else:
                self.r_min_1 = 0

            self.use_mask_2 = st.checkbox(
                f"Włącz maskę dla sektora 2 ({a2_deg}°)", value=False
            )
            if self.use_mask_2:
                self.r_min_2 = st.slider(
                    "Maska od środka - sektor 2",
                    0,
                    max_mask_limit,
                    default_rmin,
                )
            else:
                self.r_min_2 = 0

        # 5. Parametry analizy
        with st.sidebar.expander("🧪 Parametry analizy", expanded=True):
            st.caption(
                "Zakres Q, korekta tła i sposób normalizacji użyte do obliczenia funkcji w przestrzeni Q i G(r)."
            )

            self.q_min = st.number_input("Q min (Å⁻¹)", value=2.0, format="%.4f")
            self.q_max = st.number_input("Q max (Å⁻¹)", value=9.0, format="%.4f")

            self.signal_mode = st.radio(
                "Tryb normalizacji",
                ["Relative contrast", "Approximate S(Q)"],
                help=(
                    "Relative contrast: bezpieczniejszy metodologicznie do porównań między sektorami. "
                    "Approximate S(Q): przybliżone S(Q), zakotwiczone na wysokim Q; interpretować ostrożnie ilościowo."
                ),
            )

            self.bg_method = st.radio(
                "Metoda tła",
                ["Gauss (wygładzanie)", "ALS (baseline)"],
                help=(
                    "Gauss: tło = wygładzona wersja danych I(Q) - szybkie, ale dla jednego szerokiego garbu "
                    "może częściowo śledzić sam pik. "
                    "ALS: dopasowuje gładką linię bazową pod pikami, zwykle lepszą dla szerokich maksimów."
                ),
            )

            self.bg_offset = st.slider(
                "Korekta tła (Background Offset)", 0.7, 1.3, 0.92
            )

            if self.bg_method.startswith("Gauss"):
                self.bg_sigma_frac = st.slider(
                    "Wygładzanie tła (% liczby punktów Q)",
                    1,
                    50,
                    6,
                    help=(
                        "Sigma filtra Gaussa podana jako % liczby punktów w wybranym zakresie Q. "
                        "Mniejsza wartość = tło bliżej I(Q), większa = silniejsze wygładzenie."
                    ),
                ) / 100.0
            else:
                self.als_log_lam = st.slider(
                    "log10(λ) - gładkość linii bazowej ALS",
                    2.0,
                    8.0,
                    5.0,
                    step=0.5,
                    help="Większa wartość = bardziej sztywna/gładka linia bazowa.",
                )
                self.als_p = st.number_input(
                    "Asymetria p (ALS)",
                    min_value=0.001,
                    max_value=0.5,
                    value=0.01,
                    step=0.001,
                    format="%.3f",
                    help=(
                        "Mała wartość = baseline silniej przyciągany do dolnej obwiedni danych."
                    ),
                )

            self.high_q_frac = st.slider(
                "Zakres high-Q do normalizacji (%)",
                10,
                40,
                20,
                help=(
                    "Końcowy procent zakresu Q używany do zakotwiczenia poziomu funkcji "
                    "w obszarze wysokiego Q."
                ),
            ) / 100.0

            self.lorch_strength = st.slider(
                "Siła okna Lorcha (efektywny Qmax / Qmax)",
                0.3,
                1.0,
                1.0,
                step=0.05,
                help=(
                    "1.0 = standardowe okno Lorcha zerujące się w Q max. "
                    "Mniejsze wartości dają silniejsze tłumienie wysokiego Q."
                ),
            )

            self.peak_search_r_min = st.number_input(
                "Minimalne r do wykrywania pików (Å)",
                min_value=0.0,
                value=2.2,
                step=0.1,
                format="%.2f",
            )

            self.peak_min_distance = st.number_input(
                "Minimalna odległość między pikami (Å)",
                min_value=0.05,
                value=0.30,
                step=0.05,
                format="%.2f",
            )

            self.peak_prominence_frac = st.slider(
                "Próg wykrywania pików (% zakresu G(r))",
                1,
                30,
                8,
                help="Próg prominence liczony względem aktualnego zakresu G(r).",
            ) / 100.0

        self.r_vals = np.linspace(0.01, 10.0, 1000)

    @staticmethod
    def _in_sector(ang, target, half_width):
        diff = (ang - target + 180) % 360 - 180
        return np.abs(diff) < half_width

    def _compute_geometry(self):
        y, x = np.indices(self.image.shape)
        dx = x - self.center[0]
        dy = y - self.center[1]
        r_dist = np.sqrt(dx**2 + dy**2)
        raw_angles = np.degrees(np.arctan2(dy, dx)) % 360
        return x, y, r_dist, raw_angles

    def _build_background(self, i):
        if self.bg_method.startswith("Gauss"):
            sigma_px = max(3.0, self.bg_sigma_frac * len(i))
            bg = gaussian_filter1d(i, sigma_px) * self.bg_offset
        else:
            lam = 10.0 ** self.als_log_lam
            bg = als_baseline(i, lam=lam, p=self.als_p) * self.bg_offset
        return np.asarray(bg, dtype=float)

    def _high_q_mask(self, q):
        q_span = self.q_max - self.q_min
        hq_start = self.q_max - self.high_q_frac * q_span
        return q >= hq_start

    def _validate_background(self, bg):
        frac_nonpositive = np.mean(bg <= 0)
        if frac_nonpositive > 0.05:
            raise ValueError(
                "Model tła daje zbyt wiele wartości <= 0. Zmień parametry tła albo zakres Q."
            )

    def process(self):
        if self.q_min >= self.q_max:
            raise ValueError("Q min musi być mniejsze niż Q max.")

        _, _, r_dist, raw_angles = self._compute_geometry()

        a1 = self.target_angle % 360
        a2 = (self.target_angle + 180) % 360

        mask_sector_1 = self._in_sector(
            raw_angles, a1, self.sector_half_width
        ) & (r_dist >= self.r_min_1)
        mask_sector_2 = self._in_sector(
            raw_angles, a2, self.sector_half_width
        ) & (r_dist >= self.r_min_2)
        mask = mask_sector_1 | mask_sector_2

        if not np.any(mask):
            raise ValueError(
                "Wybrany sektor nie zawiera żadnych pikseli. Zmień kąt, szerokość lub maskę wewnętrzną."
            )

        r_masked = r_dist[mask]
        valid_pixels = self.image[mask]

        if r_masked.size == 0 or valid_pixels.size == 0:
            raise ValueError("Brak danych po zastosowaniu maski sektora.")

        # Prosty radialny binning 1 px - zostawiony celowo, żeby zmiany były minimalne.
        r_int = r_masked.astype(int)

        max_r = int(np.max(r_int))
        counts = np.bincount(r_int, minlength=max_r + 1)
        weighted = np.bincount(r_int, weights=valid_pixels, minlength=max_r + 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            i_raw = weighted / counts

        i_raw = np.nan_to_num(i_raw, nan=0.0, posinf=0.0, neginf=0.0)

        q_full = np.arange(len(i_raw), dtype=float) * self.q_per_pixel
        m = (q_full >= self.q_min) & (q_full <= self.q_max)

        if not np.any(m):
            raise ValueError(
                "Zakres Q min / Q max nie zawiera żadnych punktów. Sprawdź skalę q/pixel i zakres Q."
            )

        q = q_full[m]
        i = i_raw[m]

        if q.size < 5:
            raise ValueError("Za mało punktów w wybranym zakresie Q do stabilnej analizy.")

        bg = self._build_background(i)
        self._validate_background(bg)

        bg_safe = np.maximum(bg, 1e-12)

        hq_mask = self._high_q_mask(q)
        if np.count_nonzero(hq_mask) < 3:
            raise ValueError(
                "Za mało punktów w obszarze high-Q do stabilnej normalizacji."
            )

        if self.signal_mode == "Relative contrast":
            q_signal = (i - bg_safe) / bg_safe
            hq_offset = np.mean(q_signal[hq_mask])

            if not np.isfinite(hq_offset):
                raise ValueError("Nie udało się wyznaczyć poprawnego offsetu high-Q.")

            q_signal = q_signal - hq_offset
            signal_label = "C(Q)"
        else:
            ratio = i / bg_safe
            hq_mean = np.mean(ratio[hq_mask])

            if not np.isfinite(hq_mean) or abs(hq_mean) < 1e-12:
                raise ValueError("Nie udało się wyznaczyć poprawnej normalizacji S(Q).")

            q_signal = ratio / hq_mean
            signal_label = "S(Q)"

        q_signal = np.nan_to_num(q_signal, nan=0.0, posinf=0.0, neginf=0.0)

        lorch_qmax = max(self.lorch_strength * self.q_max, self.q_min + 1e-9)
        x = np.clip(q / lorch_qmax, 0.0, 1.0)
        lorch = np.where(q <= lorch_qmax, np.sinc(x), 0.0)

        if self.signal_mode == "Relative contrast":
            integrand = (
                q[:, None]
                * q_signal[:, None]
                * np.sin(q[:, None] * self.r_vals)
                * lorch[:, None]
            )
        else:
            integrand = (
                q[:, None]
                * (q_signal[:, None] - 1.0)
                * np.sin(q[:, None] * self.r_vals)
                * lorch[:, None]
            )

        g_r = (2.0 / np.pi) * _trapezoid(integrand, q, axis=0)
        g_r = np.nan_to_num(g_r, nan=0.0, posinf=0.0, neginf=0.0)

        high_q_std = float(np.std(q_signal[hq_mask]))

        return {
            "q": q,
            "i": i,
            "bg": bg,
            "q_signal": q_signal,
            "g_r": g_r,
            "signal_label": signal_label,
            "high_q_std": high_q_std,
            "hq_mask": hq_mask,
        }

    def run(self):
        result = self.process()

        q = result["q"]
        i = result["i"]
        bg = result["bg"]
        q_signal = result["q_signal"]
        g_r = result["g_r"]
        signal_label = result["signal_label"]
        high_q_std = result["high_q_std"]

        if high_q_std > 0.15:
            st.warning(
                "Obszar high-Q nie wygląda na bardzo stabilny. Wynik może być silnie zależny od parametrów tła i normalizacji."
            )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dyfrakcja i sektory")
            fig1, ax1 = plt.subplots(figsize=(6.4, 4.8))

            display_img = self.image.copy()
            _, _, r_dist, raw_angles = self._compute_geometry()

            a1 = self.target_angle % 360
            a2 = (self.target_angle + 180) % 360

            in_masked_zone_1 = self._in_sector(
                raw_angles, a1, self.sector_half_width
            ) & (r_dist < self.r_min_1)
            in_masked_zone_2 = self._in_sector(
                raw_angles, a2, self.sector_half_width
            ) & (r_dist < self.r_min_2)

            display_img[in_masked_zone_1 | in_masked_zone_2] = 0

            ax1.imshow(
                np.log10(np.clip(display_img, a_min=0, a_max=None) + 1),
                cmap="magma",
            )
            ax1.plot(self.center[0], self.center[1], "r+", ms=15, mew=2)

            if self.show_guide_circle:
                guide_circle = Circle(
                    (self.center[0], self.center[1]),
                    self.guide_radius,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.5,
                    linestyle="-",
                    alpha=0.9,
                )
                ax1.add_patch(guide_circle)

            line_length = self.crop_val

            angle_rmin_pairs = [
                (self.target_angle - self.sector_half_width, self.r_min_1),
                (self.target_angle + self.sector_half_width, self.r_min_1),
                (self.target_angle + 180 - self.sector_half_width, self.r_min_2),
                (self.target_angle + 180 + self.sector_half_width, self.r_min_2),
            ]

            for ang, r_min in angle_rmin_pairs:
                rad = np.radians(ang)

                x_start = self.center[0] + r_min * np.cos(rad)
                y_start = self.center[1] + r_min * np.sin(rad)

                x_end = self.center[0] + line_length * np.cos(rad)
                y_end = self.center[1] + line_length * np.sin(rad)

                ax1.plot([x_start, x_end], [y_start, y_end], color="cyan", lw=1.2)

            ax1.set_title("Obraz dyfrakcyjny z sektorami")
            ax1.set_axis_off()
            st.pyplot(fig1)

        with col2:
            st.subheader("Profil Q i funkcja G(r)")
            fig2, axes = plt.subplots(3, 1, figsize=(7.2, 8.8), sharex=False)

            ax2, ax3, ax4 = axes

            ax2.plot(q, i, label="I(Q)", color="black", lw=1.2)
            ax2.plot(q, bg, label="Background", color="tab:orange", lw=1.2)
            ax2.set_ylabel("Intensywność")
            ax2.set_title("Profil sektorowy I(Q) i tło")
            ax2.legend()
            ax2.grid(alpha=0.25)

            ax3.plot(q, q_signal, color="tab:blue", lw=1.2, label=signal_label)
            if signal_label == "S(Q)":
                ax3.axhline(1.0, color="gray", ls="--", lw=1.0)
            else:
                ax3.axhline(0.0, color="gray", ls="--", lw=1.0)
            ax3.set_ylabel(signal_label)
            ax3.set_title(f"{signal_label} po normalizacji high-Q")
            ax3.legend()
            ax3.grid(alpha=0.25)

            ax4.plot(self.r_vals, g_r, color="tab:red", lw=1.3, label="G(r)")
            ax4.axhline(0.0, color="gray", ls="--", lw=1.0)

            gr_range = float(np.max(g_r) - np.min(g_r))
            prominence = max(1e-12, self.peak_prominence_frac * gr_range)

            mask_r = self.r_vals > self.peak_search_r_min
            r_for_peaks = self.r_vals[mask_r]
            g_for_peaks = g_r[mask_r]

            if len(r_for_peaks) >= 3:
                dr = self.r_vals[1] - self.r_vals[0]
                distance_pts = max(1, int(self.peak_min_distance / dr))

                peaks, _ = find_peaks(
                    g_for_peaks,
                    prominence=prominence,
                    distance=distance_pts,
                )

                if peaks.size > 0:
                    peak_r = r_for_peaks[peaks]
                    peak_g = g_for_peaks[peaks]

                    ax4.plot(peak_r, peak_g, "ko", ms=4)

                    for pr, pg in zip(peak_r, peak_g):
                        ax4.annotate(
                            f"{pr:.2f} Å",
                            xy=(pr, pg),
                            xytext=(4, 6),
                            textcoords="offset points",
                            fontsize=8,
                        )

            ax4.set_xlabel("r (Å)")
            ax4.set_ylabel("G(r)")
            ax4.set_title("Funkcja w przestrzeni rzeczywistej")
            ax4.grid(alpha=0.25)

            fig2.tight_layout()
            st.pyplot(fig2)

        st.subheader("Podsumowanie parametrów")
        st.markdown(
            f"""
**Tryb analizy:** {self.signal_mode}  
**Metoda tła:** {self.bg_method}  
**Q range:** {self.q_min:.3f} - {self.q_max:.3f} Å⁻¹  
**Normalizacja high-Q:** ostatnie {int(round(self.high_q_frac * 100))}% zakresu Q  
**Kąt bazowy:** {self.target_angle}°  
**Pół-szerokość sektora:** {self.sector_half_width}°  
**Maska sektora 1:** {self.r_min_1} px  
**Maska sektora 2:** {self.r_min_2} px  
**q/pixel z pliku:** {self.q_per_pixel_file:.6f} Å⁻¹/pixel  
**Mnożnik korekcyjny:** {self.q_scale_correction:.3f}  
**q/pixel użyte w analizie:** {self.q_per_pixel:.6f} Å⁻¹/pixel  
**Odchylenie standardowe w high-Q:** {high_q_std:.4f}
"""
        )

        export_arr = np.column_stack((self.r_vals, g_r))
        export_name = (
            "relative_G_r.csv"
            if self.signal_mode == "Relative contrast"
            else "approximate_G_r.csv"
        )

        st.download_button(
            "Pobierz G(r) jako CSV",
            data=io.BytesIO(
                (
                    "r_A,G_r\n"
                    + "\n".join(f"{r:.6f},{g:.10e}" for r, g in export_arr)
                ).encode("utf-8")
            ),
            file_name=export_name,
            mime="text/csv",
        )

        q_export = np.column_stack((q, i, bg, q_signal))
        q_header = f"Q_A^-1,I_Q,Background,{signal_label}"

        st.download_button(
            "Pobierz pełny profil Q jako CSV",
            data=io.BytesIO(
                (
                    q_header + "\n"
                    + "\n".join(
                        f"{qq:.6f},{ii:.10e},{bb:.10e},{ss:.10e}"
                        for qq, ii, bb, ss in q_export
                    )
                ).encode("utf-8")
            ),
            file_name="q_profile_full.csv",
            mime="text/csv",
        )


def get_safe_scale(signal) -> float:
    scale = None

    try:
        scale = float(signal.axes_manager[0].scale)
    except Exception:
        scale = None

    if scale is None or not np.isfinite(scale) or scale <= 0:
        st.sidebar.warning(
            "Nie udało się odczytać poprawnej skali osi. Podaj ręcznie q/pixel."
        )
        scale = st.sidebar.number_input(
            "Skala q/pixel (Å⁻¹/pixel)",
            min_value=1e-6,
            value=0.01,
            format="%.6f",
        )
    else:
        if np.isclose(scale, 1.0):
            st.sidebar.info(
                "Wykryto skalę równą 1.0. Jeśli to nie jest rzeczywista kalibracja q/pixel, podaj poprawną wartość ręcznie."
            )
            use_detected = st.sidebar.checkbox(
                "Użyj wykrytej skali 1.0", value=False
            )
            if not use_detected:
                scale = st.sidebar.number_input(
                    "Skala q/pixel (Å⁻¹/pixel)",
                    min_value=1e-6,
                    value=0.01,
                    format="%.6f",
                )

    return float(scale)


def load_data(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        signal = hs.load(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    data = signal.data
    scale = get_safe_scale(signal)
    return data, scale


def main():
    st.title("PDF Analyzer")
    st.caption(
        "Sektorowa analiza dyfrakcji 2D z obliczaniem Relative G(r) lub przybliżonego S(Q)."
    )

    uploaded_file = st.file_uploader(
        "Wczytaj plik z danymi dyfrakcyjnymi",
        type=None,
    )

    if uploaded_file is None:
        st.info("Wybierz plik, aby rozpocząć analizę.")
        return

    try:
        raw_data, q_per_pixel = load_data(uploaded_file)
        analyzer = PDFWebAnalyzer(raw_data, q_per_pixel)
        analyzer.run()
    except Exception as e:
        st.error(f"Błąd: {e}")


if __name__ == "__main__":
    main()
