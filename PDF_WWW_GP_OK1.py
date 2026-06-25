import io
import os
import tempfile

import numpy as np
import streamlit as st
import hyperspy.api as hs
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.integrate import trapezoid


st.set_page_config(page_title="PDF Analyzer", layout="wide")


class PDFWebAnalyzer:
    def __init__(self, raw_data: np.ndarray, pixel_size: float):
        self.raw_data = np.asarray(raw_data, dtype=float)
        self.pixel_size = float(pixel_size)

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

        # 3. Sektor badawczy
        with st.sidebar.expander("🍕 Sektor badawczy", expanded=True):
            st.caption("Wybierz kierunek analizy i szerokość sektorów.")

            self.target_angle = st.slider("Kąt bazowy (Target Angle)", 0, 360, 135)
            self.sector_width = st.slider("Szerokość sektora (°)", 1, 90, 15)

        # 4. Maskowanie skrzydeł
        with st.sidebar.expander("📐 Maskowanie skrzydeł", expanded=True):
            st.caption("Wytnij wewnętrzne części sektorów, np. obszar przesłony.")

            max_mask_limit = max(1, self.crop_val // 2)

            a1_deg = self.target_angle % 360
            a2_deg = (self.target_angle + 180) % 360

            default_rmin = min(200, max_mask_limit // 2 if max_mask_limit < 200 else 200)

            self.use_mask_1 = st.checkbox(
                f"Włącz maskę dla Skrzydła 1 ({a1_deg}°)", value=False
            )
            if self.use_mask_1:
                self.r_min_1 = st.slider(
                    "Maska od środka - Skrzydło 1",
                    0,
                    max_mask_limit,
                    default_rmin,
                )
            else:
                self.r_min_1 = 0

            self.use_mask_2 = st.checkbox(
                f"Włącz maskę dla Skrzydła 2 ({a2_deg}°)", value=False
            )
            if self.use_mask_2:
                self.r_min_2 = st.slider(
                    "Maska od środka - Skrzydło 2",
                    0,
                    max_mask_limit,
                    default_rmin,
                )
            else:
                self.r_min_2 = 0

        # 5. Parametry PDF
        with st.sidebar.expander("🧪 Parametry PDF", expanded=True):
            st.caption("Zakres Q i korekta tła użyte do obliczenia S(Q) oraz G(r).")

            self.q_min = st.number_input("Q min (Å⁻¹)", value=2.0, format="%.4f")
            self.q_max = st.number_input("Q max (Å⁻¹)", value=9.0, format="%.4f")
            self.bg_offset = st.slider(
                "Korekta tła (Background Offset)", 0.7, 1.3, 0.92
            )

        self.r_vals = np.linspace(0.01, 10.0, 1000)

    @staticmethod
    def _in_sector(ang, target, width):
        diff = (ang - target + 180) % 360 - 180
        return np.abs(diff) < width

    def process(self):
        if self.q_min >= self.q_max:
            raise ValueError("Q min musi być mniejsze niż Q max.")

        y, x = np.indices(self.image.shape)
        dx = x - self.center[0]
        dy = y - self.center[1]
        r_dist = np.sqrt(dx**2 + dy**2)
        raw_angles = np.degrees(np.arctan2(dy, dx)) % 360

        a1 = self.target_angle % 360
        a2 = (self.target_angle + 180) % 360

        mask_skrzydlo_1 = self._in_sector(raw_angles, a1, self.sector_width) & (r_dist >= self.r_min_1)
        mask_skrzydlo_2 = self._in_sector(raw_angles, a2, self.sector_width) & (r_dist >= self.r_min_2)
        mask = mask_skrzydlo_1 | mask_skrzydlo_2

        if not np.any(mask):
            raise ValueError(
                "Wybrany sektor nie zawiera żadnych pikseli. Zmień kąt, szerokość lub maskę wewnętrzną."
            )

        r_masked = r_dist[mask]
        valid_pixels = self.image[mask]

        if r_masked.size == 0 or valid_pixels.size == 0:
            raise ValueError("Brak danych po zastosowaniu maski sektora.")

        r_int = r_masked.astype(int)

        max_r = int(np.max(r_int))
        counts = np.bincount(r_int, minlength=max_r + 1)
        weighted = np.bincount(r_int, weights=valid_pixels, minlength=max_r + 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            i_raw = weighted / counts

        i_raw = np.nan_to_num(i_raw, nan=0.0, posinf=0.0, neginf=0.0)

        q_full = np.arange(len(i_raw), dtype=float) * self.pixel_size
        m = (q_full >= self.q_min) & (q_full <= self.q_max)

        if not np.any(m):
            raise ValueError(
                "Zakres Q min / Q max nie zawiera żadnych punktów. Sprawdź skalę piksela i zakres Q."
            )

        q = q_full[m]
        i = i_raw[m]

        if q.size < 5:
            raise ValueError("Za mało punktów w wybranym zakresie Q do stabilnej analizy.")

        bg = gaussian_filter1d(i, 60) * self.bg_offset
        bg = np.where(np.abs(bg) < 1e-12, 1e-12, bg)

        ratio = i / bg
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
        s_q = 1.0 + (ratio - np.mean(ratio))

        lorch = np.sinc(q * np.pi / self.q_max)
        integrand = (
            q[:, None]
            * (s_q[:, None] - 1.0)
            * np.sin(q[:, None] * self.r_vals)
            * lorch[:, None]
        )
        g_r = (2.0 / np.pi) * trapezoid(integrand, q, axis=0)

        g_r = np.nan_to_num(g_r, nan=0.0, posinf=0.0, neginf=0.0)

        return q, i, bg, s_q, g_r

    def run(self):
        q, i, bg, s_q, g_r = self.process()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dyfrakcja i sektory")
            fig1, ax1 = plt.subplots(figsize=(6, 6))

            display_img = self.image.copy()
            y, x = np.indices(display_img.shape)
            r_dist = np.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2)
            raw_angles = np.degrees(
                np.arctan2(y - self.center[1], x - self.center[0])
            ) % 360

            a1 = self.target_angle % 360
            a2 = (self.target_angle + 180) % 360

            in_masked_zone_1 = self._in_sector(raw_angles, a1, self.sector_width) & (
                r_dist < self.r_min_1
            )
            in_masked_zone_2 = self._in_sector(raw_angles, a2, self.sector_width) & (
                r_dist < self.r_min_2
            )

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
                (self.target_angle - self.sector_width, self.r_min_1),
                (self.target_angle + self.sector_width, self.r_min_1),
                (self.target_angle + 180 - self.sector_width, self.r_min_2),
                (self.target_angle + 180 + self.sector_width, self.r_min_2),
            ]

            for ang, r_min in angle_rmin_pairs:
                rad = np.radians(ang)

                x_end = self.center[0] + line_length * np.cos(rad)
                y_end = self.center[1] + line_length * np.sin(rad)

                x_start = self.center[0] + r_min * np.cos(rad)
                y_start = self.center[1] + r_min * np.sin(rad)

                ax1.plot([x_start, x_end], [y_start, y_end], "w--", alpha=0.6, lw=1)

                if r_min > 0:
                    ax1.plot(x_start, y_start, "g.", ms=8)

            ax1.axis("off")
            fig1.tight_layout()
            st.pyplot(fig1)

            st.subheader("Czynnik strukturalny S(Q)")
            fig2, ax2 = plt.subplots()
            ax2.plot(q, s_q, color="darkgreen")
            ax2.axhline(1.0, color="k", ls="--")
            ax2.set_xlabel("Q (Å⁻¹)")
            ax2.set_ylabel("S(Q)")
            ax2.grid(alpha=0.2)
            st.pyplot(fig2)

        with col2:
            st.subheader("Profil I(Q)")
            fig3, ax3 = plt.subplots()
            ax3.plot(q, i, label="Eksperyment")
            ax3.plot(q, bg, "r--", label="Tło")
            ax3.set_xlabel("Q (Å⁻¹)")
            ax3.set_ylabel("I(Q)")
            ax3.legend()
            ax3.grid(alpha=0.2)
            st.pyplot(fig3)

            st.subheader("PDF G(r)")
            fig4, ax4 = plt.subplots()
            ax4.plot(self.r_vals, g_r, color="blue", lw=1.5)

            mask_r = self.r_vals > 2.2
            if np.any(mask_r):
                p_idx, _ = find_peaks(g_r[mask_r], prominence=0.03, distance=40)
                real_idx = np.where(mask_r)[0][p_idx]

                for idx in real_idx:
                    ax4.plot(self.r_vals[idx], g_r[idx], "ro")
                    ax4.text(
                        self.r_vals[idx],
                        g_r[idx] + 0.02,
                        f"{self.r_vals[idx]:.2f}",
                        ha="center",
                        color="red",
                    )

            ax4.set_xlabel("r (Å)")
            ax4.set_ylabel("G(r)")
            ax4.set_xlim(0, 10)

            ax4.xaxis.set_major_locator(MultipleLocator(1.0))
            ax4.xaxis.set_minor_locator(MultipleLocator(0.1))

            ax4.grid(which="major", axis="x", alpha=0.55, linewidth=0.9)
            ax4.grid(which="minor", axis="x", alpha=0.18, linewidth=0.5)
            ax4.grid(which="major", axis="y", alpha=0.2, linewidth=0.8)

            st.pyplot(fig4)

        st.divider()

        output = io.StringIO()
        np.savetxt(
            output,
            np.column_stack((self.r_vals, g_r)),
            header="r(A) G(r)",
            fmt="%.6e",
        )
        st.download_button(
            "Pobierz wynik G(r) (.txt)",
            output.getvalue(),
            "final_pdf.txt",
            mime="text/plain",
        )


def get_safe_scale(signal) -> float:
    try:
        scale = signal.axes_manager[0].scale
        if scale is None or not np.isfinite(scale) or scale <= 0:
            return 1.0
        return float(scale)
    except (AttributeError, IndexError, TypeError):
        return 1.0


def save_uploaded_files(uploaded_files, temp_dir):
    emi_path = None
    emd_path = None
    first_path = None

    for uploaded_file in uploaded_files:
        f_path = os.path.join(temp_dir, uploaded_file.name)

        with open(f_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if first_path is None:
            first_path = f_path

        lower_name = uploaded_file.name.lower()
        if lower_name.endswith(".emi"):
            emi_path = f_path
        elif lower_name.endswith(".emd"):
            emd_path = f_path

    if emd_path:
        return emd_path
    if emi_path:
        return emi_path
    return first_path


def clear_session_files(temp_dir):
    if not os.path.exists(temp_dir):
        return

    for fname in os.listdir(temp_dir):
        fpath = os.path.join(temp_dir, fname)
        try:
            if os.path.isfile(fpath):
                os.remove(fpath)
        except Exception as e:
            st.warning(f"Nie udało się usunąć {fname}: {e}")


def main():
    st.title("🔬 PDF Analyzer")
    st.caption("Analiza danych dyfrakcyjnych i wyznaczanie G(r) z sektorów obrazu 2D.")

    if "session_dir" not in st.session_state:
        st.session_state.session_dir = tempfile.mkdtemp(prefix="pdf_analysis_")

    temp_dir = st.session_state.session_dir

    with st.sidebar.expander("🧹 Sesja", expanded=False):
        if st.button("Wyczyść pliki sesji"):
            clear_session_files(temp_dir)
            st.success("Wyczyszczono pliki sesji.")

    uploaded_files = st.file_uploader(
        "Wgraj pliki",
        type=["emi", "ser", "dm3", "dm4", "hspy", "tiff", "emd"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Wgraj co najmniej jeden plik, aby rozpocząć analizę.")
        return

    load_target = save_uploaded_files(uploaded_files, temp_dir)

    if not load_target:
        st.error("Nie udało się przygotować pliku do analizy.")
        return

    try:
        data_loaded = hs.load(load_target)

        if isinstance(data_loaded, list):
            st.info(f"Plik zawiera {len(data_loaded)} obiektów. Używam pierwszego.")
            if len(data_loaded) == 0:
                st.error("Plik nie zawiera danych do analizy.")
                return
            data_loaded = data_loaded[0]

        if not hasattr(data_loaded, "data"):
            st.error("Załadowany obiekt nie zawiera pola 'data'.")
            return

        scale = get_safe_scale(data_loaded)

        if scale <= 0 or np.isclose(scale, 1.0):
            scale = st.sidebar.number_input(
                "Skala piksela (1/Å):",
                min_value=1e-6,
                value=0.02,
                format="%.6f",
            )

        analyzer = PDFWebAnalyzer(data_loaded.data, scale)
        analyzer.run()

    except Exception as e:
        st.error(f"Błąd podczas analizy: {e}")


if __name__ == "__main__":
    main()
