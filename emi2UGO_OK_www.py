import streamlit as st
import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, gaussian_filter1d
from scipy.signal import find_peaks
import io
import os
import shutil

# Konfiguracja strony
st.set_page_config(page_title="PDF Analyzer", layout="wide")

class PDFWebAnalyzer:
    def __init__(self, raw_data, pixel_size):
        self.raw_data = raw_data
        self.pixel_size = pixel_size
        
        st.sidebar.header("⚙️ Ustawienia Analizy")
        
        # 1. Wycinanie i środek
        h, w = raw_data.shape
        max_c = min(h, w) // 2
        c = st.sidebar.slider("Rozmiar wycinka (crop)", 50, max_c, min(450, max_c))
        
        self.image = raw_data[h//2-c:h//2+c, w//2-c:w//2+c]
        
        # Automatyczne wykrywanie środka (na bazie oryginalnej logiki)
        threshold = np.percentile(self.image, 99.7)
        mask_init = self.image > threshold
        com = center_of_mass(mask_init)
        
        st.sidebar.subheader("📍 Pozycjonowanie środka")
        off_x = st.sidebar.number_input("Przesunięcie X", value=0.0, step=0.5)
        off_y = st.sidebar.number_input("Przesunięcie Y", value=0.0, step=0.5)
        self.center = [com[1] + off_x, com[0] + off_y]

        # 2. Sektor (odpowiednik kółka myszy z oryginału)
        st.sidebar.subheader("🍕 Sektor")
        self.target_angle = st.sidebar.slider("Kąt (Target Angle)", 0, 360, 135)
        self.sector_width = st.sidebar.slider("Szerokość sektora (°)", 1, 90, 15)
        
        # 3. Parametry fizyczne
        st.sidebar.subheader("🧪 Parametry PDF")
        self.q_min = st.sidebar.number_input("Q min (Å⁻¹)", value=1.6)
        self.q_max = st.sidebar.number_input("Q max (Å⁻¹)", value=10.5)
        self.bg_offset = st.sidebar.slider("Korekta tła (Background Offset)", 0.7, 1.3, 0.92)
        
        self.r_vals = np.linspace(0.01, 10.0, 1000)

    def process(self):
        y, x = np.indices(self.image.shape)
        dx, dy = x - self.center[0], y - self.center[1]
        r_dist = np.sqrt(dx**2 + dy**2)
        raw_angles = np.degrees(np.arctan2(dy, dx)) % 360
        
        a1, a2 = self.target_angle % 360, (self.target_angle + 180) % 360
        
        def in_sector(ang, target, w):
            diff = (ang - target + 180) % 360 - 180
            return np.abs(diff) < w

        mask = in_sector(raw_angles, a1, self.sector_width) | \
               in_sector(raw_angles, a2, self.sector_width)
        
        r_int = r_dist[mask].astype(int)
        valid_pixels = self.image[mask]
        
        max_r = int(np.max(r_dist))
        counts = np.bincount(r_int.ravel(), minlength=max_r+1)
        counts[counts == 0] = 1
        i_raw = np.bincount(r_int.ravel(), valid_pixels.ravel(), minlength=len(counts)) / counts
        
        q_full = np.arange(len(i_raw)) * self.pixel_size
        m = (q_full >= self.q_min) & (q_full <= self.q_max)
        q, i = q_full[m], i_raw[m]

        bg = gaussian_filter1d(i, 60) * self.bg_offset
        s_q = 1.0 + ((i / bg) - np.mean(i / bg)) 
        
        lorch = np.sinc(q * np.pi / self.q_max)
        integrand = q[:, None] * (s_q[:, None] - 1) * np.sin(q[:, None] * self.r_vals) * lorch[:, None]
        g_r = (2 / np.pi) * np.trapezoid(integrand, q, axis=0)
        
        return q, i, bg, s_q, g_r

    def run(self):
        q, i, bg, s_q, g_r = self.process()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dyfrakcja i Sektory")
            fig1, ax1 = plt.subplots()
            ax1.imshow(np.log10(self.image + 1), cmap='magma')
            ax1.plot(self.center[0], self.center[1], 'r+', ms=15, mew=2)
            
            for ang in [self.target_angle - self.sector_width, self.target_angle + self.sector_width,
                        self.target_angle + 180 - self.sector_width, self.target_angle + 180 + self.sector_width]:
                rad = np.radians(ang)
                ax1.plot([self.center[0], self.center[0] + 500*np.cos(rad)],
                         [self.center[1], self.center[1] + 500*np.sin(rad)], 'w--', alpha=0.5, lw=1)
            ax1.axis('off')
            st.pyplot(fig1)

            st.subheader("Czynnik strukturalny S(Q)")
            fig2, ax2 = plt.subplots()
            ax2.plot(q, s_q, color='darkgreen')
            ax2.axhline(1, color='k', ls='--')
            ax2.set_xlabel("Q (Å⁻¹)")
            st.pyplot(fig2)

        with col2:
            st.subheader("Profil I(Q)")
            fig3, ax3 = plt.subplots()
            ax3.plot(q, i, label='Eksperyment')
            ax3.plot(q, bg, 'r--', label='Tło')
            ax3.set_xlabel("Q (Å⁻¹)")
            ax3.legend()
            st.pyplot(fig3)

            st.subheader("PDF G(r)")
            fig4, ax4 = plt.subplots()
            ax4.plot(self.r_vals, g_r, color='blue', lw=1.5)
            
            mask_r = self.r_vals > 2.2
            p_idx, _ = find_peaks(g_r[mask_r], prominence=0.03, distance=40)
            real_idx = np.where(mask_r)[0][p_idx]
            for idx in real_idx:
                ax4.plot(self.r_vals[idx], g_r[idx], "ro")
                ax4.text(self.r_vals[idx], g_r[idx] + 0.02, f"{self.r_vals[idx]:.2f}", ha='center', color='red')
            
            ax4.set_xlabel("r (Å)")
            ax4.set_xlim(0, 10)
            st.pyplot(fig4)

        st.divider()
        output = io.StringIO()
        np.savetxt(output, np.column_stack((self.r_vals, g_r)), header="r(A) G(r)", fmt='%.6e')
        st.download_button("Pobierz wynik G(r) (.txt)", output.getvalue(), "final_pdf.txt")

# --- Logika ładowania plików ---
st.title("🔬 PDF Analyzer")
uploaded_files = st.file_uploader("Wgraj pliki (.emi i .ser razem)", type=['emi', 'ser', 'dm3', 'dm4', 'hspy', 'tiff'], accept_multiple_files=True)

if uploaded_files:
    temp_dir = "temp_analysis"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    emi_path = None
    for uploaded_file in uploaded_files:
        f_path = os.path.join(temp_dir, uploaded_file.name)
        with open(f_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if uploaded_file.name.lower().endswith(".emi"):
            emi_path = f_path

    load_target = emi_path if emi_path else (os.path.join(temp_dir, uploaded_files[0].name) if uploaded_files else None)

    if load_target:
        try:
            d = hs.load(load_target)
            if isinstance(d, list): d = d[0]
            
            try:
                scale = d.axes_manager[0].scale
            except:
                scale = 1.0

            if scale == 1.0 or scale == 0:
                scale = st.sidebar.number_input("Skala piksela (1/Å):", value=0.02, format="%.4f")
            
            analyzer = PDFWebAnalyzer(d.data.astype(float), scale)
            analyzer.run()
        except Exception as e:
            st.error(f"Błąd: {e}")
else:
    st.info("Przeciągnij pliki .emi i .ser tutaj.")
