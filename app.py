import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Nhớ thêm thư viện này
import shap
import matplotlib.pyplot as plt
import requests
from datetime import datetime, date, timedelta
import folium
from streamlit_folium import st_folium
import pycountry
st.set_page_config(page_title="Weather Forecast App", layout="wide", page_icon="🌤️")
# --- THÊM HÀM NÀY VÀO ĐẦU FILE (Sau phần import) ---
def render_header():
    # Lấy ngày hôm nay để hiển thị
    today = date.today().strftime("Ngày %d tháng %m năm %Y")
    
    st.markdown(f"""
    <style>
        .header-container {{
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }}
        .header-title {{
            font-size: 50px;
            font-weight: bold;
            margin: 0;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}
        .header-subtitle {{
            font-size: 18px;
            font-style: italic;
            opacity: 0.8;
            margin-top: 5px;
        }}
        .header-date {{
            margin-top: 15px;
            font-size: 14px;
            background-color: rgba(255,255,255,0.2);
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
        }}
    </style>
    
    <div class="header-container">
        <div class="header-title">🌤️ Dự Báo Thời Tiết AI</div>
        <div class="header-subtitle">Phân tích dữ liệu khí tượng & Mô hình học máy</div>
        <div class="header-date">📅 Hôm nay: {today}</div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model_system():
    try:
        # Load file model đã lưu
        data = joblib.load("model.pkl")
        return data["pipeline"], data["explainer"], data["feature_names"]
    except FileNotFoundError:
        st.error("❌ Lỗi: Không tìm thấy file 'model.pkl")
        st.stop()
def get_location_name(lat, lon):
    """
    Dùng Nominatim API (OpenStreetMap) để lấy tên địa điểm từ tọa độ.
    Tham số 'accept-language=en' giúp trả về tên tiếng Anh/Latin.
    """
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10&accept-language=en"
    # OpenStreetMap yêu cầu phải có User-Agent để tránh bị chặn
    headers = {'User-Agent': 'MyWeatherApp/1.0'}
    try:
        response = requests.get(url, headers=headers).json()
        # Lấy tên hiển thị đầy đủ
        address = response.get('address', {})
        city = address.get('city') or address.get('town') or address.get('village') or address.get('county') or "Unknown Location"
        country = address.get('country', '')
        return f"{city}, {country}"
    except:
        return "Vị trí không xác định"
def map_cloud_cover(percent):
    """Chuyển đổi % mây (số) sang danh mục (chữ) cho Model"""
    if percent < 10: return 'clear'
    elif percent < 40: return 'partly cloudy'
    elif percent < 80: return 'cloudy'
    else: return 'overcast'

def get_season(month):
    """Xác định mùa theo tháng"""
    if 3 <= month <= 5: return 'Spring'
    elif 6 <= month <= 8: return 'Summer'
    elif 9 <= month <= 11: return 'Autumn'
    else: return 'Winter'

def get_city_coordinates_no_key(city_name):
    """
    Dùng Geocoding API miễn phí của Open-Meteo để tìm tọa độ.
    Không cần API Key.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city_name,
        "count": 1,
        "language": "en",
        "format": "json"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            return result["latitude"], result["longitude"], result["name"], result.get("country", "")
        else:
            return None, None, None, None
    except Exception as e:
        return None, None, None, None

def get_weather_data_no_key(lat, lon, target_date):
    """
    Lấy thời tiết từ Open-Meteo.
    Tự động chọn API Lịch sử (Archive) hoặc Dự báo (Forecast) dựa vào ngày.
    """
    today = date.today()
    
    # CASE 1: DỰ BÁO (Hôm nay và Tương lai)
    # Open-Meteo Forecast cung cấp dữ liệu cho hôm nay và 16 ngày tới
    if target_date >= today:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,rain,pressure_msl,cloud_cover,visibility,wind_speed_10m,uv_index",
            "timezone": "auto",
            "start_date": target_date.strftime("%Y-%m-%d"),
            "end_date": target_date.strftime("%Y-%m-%d")
        }
        is_forecast = True

    # CASE 2: LỊCH SỬ (Quá khứ)
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,rain,pressure_msl,cloud_cover,visibility,wind_speed_10m",
            "timezone": "auto",
            "start_date": target_date.strftime("%Y-%m-%d"),
            "end_date": target_date.strftime("%Y-%m-%d")
        }
        is_forecast = False

    try:
        resp = requests.get(url, params=params)
        data = resp.json()
        
        # Kiểm tra nếu API báo lỗi (thường do quá giới hạn ngày dự báo)
        if "error" in data:
            return None, "Ngày chọn vượt quá phạm vi dữ liệu (Chỉ hỗ trợ quá khứ hoặc 14 ngày tới)."

        # API trả về dữ liệu 24h, ta lấy giờ giữa trưa (12:00) để đại diện
        hourly = data.get("hourly", {})
        idx = 12 
        
        # Xử lý an toàn nếu dữ liệu trả về bị thiếu
        if not hourly or len(hourly['temperature_2m']) < 13:
             return None, "Không đủ dữ liệu cho ngày này."

        # Mapping dữ liệu vào Dictionary
        extracted = {
            "Temperature": hourly['temperature_2m'][idx],
            "Humidity": hourly['relative_humidity_2m'][idx],
            "Wind Speed": hourly['wind_speed_10m'][idx],
            "Precipitation (%)": 90.0 if hourly['rain'][idx] > 0.5 else 0.0, # Ước lượng % mưa dựa trên lượng mưa mm
            "Atmospheric Pressure": hourly['pressure_msl'][idx],
            "Cloud Cover": map_cloud_cover(hourly['cloud_cover'][idx]),
            "Season": get_season(target_date.month),
            "Visibility (km)": (hourly['visibility'][idx] / 1000) if hourly['visibility'][idx] else 10.0,
            "Location": "inland", # Mặc định
            "UV Index": 5 # Mặc định trung bình
        }
        
        # Nếu là Forecast thì lấy UV Index chính xác hơn
        if is_forecast and 'uv_index' in hourly:
            extracted["UV Index"] = hourly['uv_index'][idx]
            
        return extracted, None
        
    except Exception as e:
        return None, f"Lỗi kết nối API: {str(e)}"
PL, explain, all_feature_names = load_model_system()
render_header()
st.markdown("""
<style>
    div[data-testid="stForm"] {
        background-color: transparent; 
        border: 2px solid #2196F3; 
        border-radius: 15px; 
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)
# --- THÊM IMPORT Ở ĐẦU FILE ---


# ==========================================
# GIAO DIỆN CHÍNH (FULL TÍNH NĂNG)
# ==========================================
st.title("🌏 Chọn Địa Điểm & Dữ Liệu")

# 1. Khởi tạo Session State (Bộ nhớ đệm)
if 'current_lat' not in st.session_state: st.session_state.current_lat = None
if 'current_lon' not in st.session_state: st.session_state.current_lon = None
if 'last_processed_click' not in st.session_state: st.session_state.last_processed_click = None
if 'current_location_label' not in st.session_state: st.session_state.current_location_label = None
if 'city_search_results' not in st.session_state: st.session_state.city_search_results = []
if 'form_vals' not in st.session_state:
    st.session_state.form_vals = {
        "Temperature": 25.0, "Humidity": 60, "Wind Speed": 10.0, "Precipitation (%)": 0.0,
        "Atmospheric Pressure": 1013.0, "UV Index": 5, "Visibility (km)": 10.0,
        "Season": "Spring", "Location": "inland", "Cloud Cover": "partly cloudy"
    }

# --- PHẦN 1: CHỌN NGÀY & NÚT CẬP NHẬT (QUAN TRỌNG) ---
col_date, col_btn = st.columns([2, 1])

with col_date:
    max_date = date.today() + timedelta(days=14)
    selected_date = st.date_input("📅 Chọn ngày", value=date.today(), max_value=max_date)

with col_btn:
    st.write("") # Khoảng trống căn lề
    st.write("") 
    # Nút này để lấy dữ liệu mới khi bạn đổi ngày (mà không cần chọn lại địa điểm)
    refresh_btn = st.button("🔄 Lấy dữ liệu ngày này", type="primary", use_container_width=True)


# --- PHẦN 2: TAB CHỌN ĐỊA ĐIỂM ---
tab_map, tab_manual = st.tabs(["🗺️ Chọn trên Bản đồ", "✍️ Nhập thủ công (Quốc gia/TP)"])

should_fetch_data = False
fetch_source = ""

# >>> TAB 1: BẢN ĐỒ
with tab_map:
    m = folium.Map(location=[16.047, 108.206], zoom_start=4, tiles="CartoDB positron")
    m.add_child(folium.LatLngPopup())
    map_output = st_folium(m, height=450, width=1200, returned_objects=["last_clicked"])

    if map_output and map_output['last_clicked']:
        current_click = map_output['last_clicked']
        # Chỉ xử lý khi click mới khác click cũ
        if current_click != st.session_state.last_processed_click:
            st.session_state.last_processed_click = current_click
            
            # Xử lý tọa độ (Fix lỗi kinh độ/vĩ độ ảo)
            raw_lat, raw_lon = current_click['lat'], current_click['lng']
            lon_click = ((raw_lon + 180) % 360) - 180
            lat_click = max(-90, min(90, raw_lat))
            
            # Lưu vào bộ nhớ
            st.session_state.current_lat = lat_click
            st.session_state.current_lon = lon_click
            
            # Lấy tên hiển thị
            loc_name = get_location_name(lat_click, lon_click)
            st.session_state.current_location_label = f"**{loc_name}**"
            
            should_fetch_data = True
            fetch_source = "map"

# >>> TAB 2: NHẬP TAY
with tab_manual:
    col_country, col_city = st.columns(2)
    with col_country:
        # Load danh sách quốc gia
        countries = sorted([(country.name, country.alpha_2) for country in pycountry.countries], key=lambda x: x[0])
        country_names = [c[0] for c in countries]
        try: default_ix = country_names.index("Viet Nam")
        except: default_ix = 0
        selected_country_name = st.selectbox("1. Quốc gia:", country_names, index=default_ix)
        selected_country_code = next(c[1] for c in countries if c[0] == selected_country_name)

    with col_city:
        city_query = st.text_input("2. Thành phố (Enter để tìm):", placeholder="VD: Ha Noi...")

    if city_query:
        search_url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": city_query, "count": 10, "language": "en", "format": "json"}
        try:
            res = requests.get(search_url, params=params).json()
            if "results" in res:
                st.session_state.city_search_results = [
                    item for item in res["results"] 
                    if item.get("country_code", "").upper() == selected_country_code
                ]
            else: st.session_state.city_search_results = []
        except: pass

    if st.session_state.city_search_results:
        options = {f"{i['name']} ({i.get('admin1','')})": i for i in st.session_state.city_search_results}
        s_key = st.selectbox("3. Kết quả:", list(options.keys()))
        
        if st.button("✅ Chọn địa điểm này"):
            chosen = options[s_key]
            st.session_state.current_lat = chosen['latitude']
            st.session_state.current_lon = chosen['longitude']
            st.session_state.current_location_label = f"**{chosen['name']}, {selected_country_name}**"
            should_fetch_data = True
            fetch_source = "manual"
    elif city_query:
        st.caption("Không tìm thấy kết quả phù hợp.")

# --- XỬ LÝ LOGIC NÚT CẬP NHẬT ---
if refresh_btn:
    if st.session_state.current_lat is not None:
        should_fetch_data = True
        fetch_source = "button"
    else:
        st.toast("⚠️ Bạn chưa chọn địa điểm nào!", icon="Vk")

# --- GỌI API LẤY DỮ LIỆU ---
if should_fetch_data:
    lat = st.session_state.current_lat
    lon = st.session_state.current_lon
    
    with st.spinner(f"Đang tải dữ liệu ngày {selected_date}..."):
        # Gọi hàm lấy thời tiết (đã sửa lỗi áp suất 860 ở bài trước)
        weather_data, err = get_weather_data_no_key(lat, lon, selected_date)
        
        if weather_data:
            st.session_state.form_vals.update(weather_data)
            
            # Thông báo
            if fetch_source == "map": st.toast("Đã cập nhật từ Bản đồ", icon="📍")
            elif fetch_source == "manual": st.toast("Đã cập nhật từ Nhập tay", icon="✍️")
            elif fetch_source == "button": st.toast(f"Đã cập nhật ngày {selected_date}", icon="🔄")
            
            st.rerun()
        else:
            st.error(f"Lỗi: {err}")


# --- THANH TRẠNG THÁI ---

st.divider()
if st.session_state.current_location_label:
    st.success(f"📍 Đang chọn: {st.session_state.current_location_label} - Dữ liệu ngày: **{selected_date.strftime('%d/%m/%Y')}**")
else:
    st.info("👈 Vui lòng chọn địa điểm (trên Bản đồ hoặc Nhập tay).")

st.divider()

# --- FORM NHẬP LIỆU (AUTO-FILL) ---
with st.form("weather_form"):
    st.subheader("Thông số môi trường")
    
    col1, col2 = st.columns(2)
    with col1:
        ss_season = st.session_state.form_vals['Season']
        season = st.selectbox("Mùa", options=['Spring', 'Summer', 'Autumn', 'Winter'], 
                            index=['Spring', 'Summer', 'Autumn', 'Winter'].index(ss_season))
        
        ss_loc = st.session_state.form_vals['Location']
        location = st.selectbox("Vị trí", options=['inland', 'mountain', 'coastal'],
                            index=['inland', 'mountain', 'coastal'].index(ss_loc))
        
        temperature = st.number_input("Nhiệt độ (°C)", min_value = -80.0, max_value = 80.0, value=float(st.session_state.form_vals['Temperature']))
        humidity = st.slider("Độ ẩm (%)", 0, 100, int(st.session_state.form_vals['Humidity']))
        
    with col2:
        pressure = st.number_input("Áp suất (hPa)", min_value = 850.0, max_value = 1110.0, value=float(st.session_state.form_vals['Atmospheric Pressure']))
        wind_speed = st.number_input("Tốc độ gió (km/h)", value=float(st.session_state.form_vals['Wind Speed']))
        precipitation = st.number_input("Khả năng mưa / Lượng mưa (%)", min_value = 0.0, max_value = 100.0, value=float(st.session_state.form_vals['Precipitation (%)']))
        
        valid_clouds = ['clear', 'partly cloudy', 'cloudy', 'overcast']
        current_cloud = st.session_state.form_vals['Cloud Cover']
        c_idx = valid_clouds.index(current_cloud) if current_cloud in valid_clouds else 1
        cloud_cover = st.selectbox("Độ che phủ mây", options=valid_clouds, index=c_idx)
    
    col3, col4 = st.columns(2)
    with col3: uv_index = st.slider("Chỉ số UV", 0, 20, int(st.session_state.form_vals['UV Index']))
    with col4: visibility = st.slider("Tầm nhìn (km)", 0.0, 100.0, float(st.session_state.form_vals['Visibility (km)']))

    submitted = st.form_submit_button("🚀 CHẠY DỰ BÁO", use_container_width=True)

if submitted:
    # (Phần xử lý Predict giữ nguyên như cũ)
    input_data = {
        "Temperature": temperature, "Humidity": humidity, "Wind Speed": wind_speed,
        "Precipitation (%)": precipitation, "Cloud Cover": cloud_cover,
        "Atmospheric Pressure": pressure, "UV Index": uv_index,
        "Season": season, "Visibility (km)": visibility, "Location": location
    }
    df_input = pd.DataFrame([input_data])
    output = PL.predict(df_input)[0]
    result_text = ['Snowy', 'Cloudy', 'Rainy', 'Sunny'][output]
    
    styles = {
        'Sunny': {'color': '#f39c12', 'icon': '☀️', 'vi': 'TRỜI NẮNG'},
        'Rainy': {'color': '#2980b9', 'icon': '🌧️', 'vi': 'TRỜI MƯA'},
        'Cloudy': {'color': '#7f8c8d', 'icon': '☁️', 'vi': 'NHIỀU MÂY'},
        'Snowy': {'color': '#ecf0f1', 'icon': '❄️', 'vi': 'CÓ TUYẾT', 'text': '#2c3e50'}
    }
    st_res = styles.get(result_text)
    text_color = st_res.get('text', 'white')
    
    st.markdown(f"""
        <div style="background-color: {st_res['color']}; padding: 30px; border-radius: 20px; text-align: center; margin-top: 20px;">
            <div style="font-size: 80px;">{st_res['icon']}</div>
            <h1 style="color: {text_color}; margin: 0;">{st_res['vi']}</h1>
        </div>
    """, unsafe_allow_html=True)
    
    input_scaled = PL.named_steps['preprocessor'].transform(df_input)
    shap_val_input = explain.shap_values(input_scaled)
    expl = shap.Explanation(values=shap_val_input[0,:,output], base_values=explain.expected_value[output], data=input_scaled[0], feature_names=all_feature_names)
    fig_w = plt.figure(figsize=(8, 6))
    shap.waterfall_plot(expl, show=False)
    st.pyplot(fig_w, use_container_width=False)
    
st.info(
    "⚠️ Đây là bản demo học thuật. Mô hình được huấn luyện trên dữ liệu nhân tạo, "
    "kết quả dự đoán mang tính minh họa, không thay thế dự báo khí tượng chính thức."
)
