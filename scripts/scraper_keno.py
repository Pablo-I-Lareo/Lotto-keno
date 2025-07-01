from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os, time

url = "https://lotostats.ro/toate-rezultatele-italia-keno-10e-20-90"
output_dir = "htmls_filas"
os.makedirs(output_dir, exist_ok=True)

options = Options()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)
driver.get(url)

# Aceptar cookies
try:
    boton = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'fc-button') and contains(., 'Consimțământ')]"))
    )
    driver.execute_script("arguments[0].click();", boton)
    print("✅ Consentimiento aceptado.")
except Exception as e:
    print(f"⚠ Error al aceptar cookies: {e}")

# Scroll general para que aparezca el div de resultados
for i in range(15):
    driver.execute_script("window.scrollBy(0, 600);")
    time.sleep(0.8)
    print(f"🌐 Scroll página {i+1}")

# Esperar al div interno con scroll
try:
    scroll_div = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.dataTables_scrollBody"))
    )
    print("✅ Scroll interno localizado.")
except:
    print("❌ No se localizó el contenedor con scroll interno. Cerrando.")
    driver.quit()
    exit()

# Función para hacer scroll y extraer filas
def extraer_filas(pagina):
    print(f"📄 Extrayendo datos de página {pagina}...")
    for i in range(12):
        y = (i + 1) * 400
        driver.execute_script("arguments[0].scrollTop = arguments[1];", scroll_div, y)
        time.sleep(0.4)

    filas = driver.find_elements(By.CSS_SELECTOR, "table#all_results tbody tr")
    print(f"🔍 Filas detectadas: {len(filas)}")

    for idx, fila in enumerate(filas):
        html = fila.get_attribute("outerHTML")
        with open(os.path.join(output_dir, f"p{pagina}_fila_{idx:03}.html"), "w", encoding="utf-8") as f:
            f.write(html)

# Página 1
extraer_filas(1)

# Ir a páginas 2 y 3
for pagina in range(1, 3):  # Solo 2 y 3
    try:
        boton_pagina = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, f"//a[@aria-controls='all_results' and text()='{pagina}']"))
        )
        driver.execute_script("arguments[0].click();", boton_pagina)
        print(f"➡️ Clic en página {pagina}")
        time.sleep(2)  # Esperar que recargue
        extraer_filas(pagina)
    except Exception as e:
        print(f"⚠ No se pudo ir a página {pagina}: {e}")

driver.quit()
print("✅ Extracción de 3 páginas completada.")
