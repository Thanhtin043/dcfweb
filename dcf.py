import streamlit as st
from vnstock import Vnstock, Listing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from IPython.display import display
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

# Thiết lập tiêu đề trang
st.set_page_config(page_title="Phân Tích Cổ Phiếu Việt Nam", layout="wide")

# Tạo tiêu đề ứng dụng
st.title("📊 Phân Tích Cổ Phiếu Việt Nam")

# Hướng dẫn sử dụng
st.markdown("""
Ứng dụng này cho phép bạn:
1. Xem dữ liệu tài chính và giá lịch sử của cổ phiếu
2. Thực hiện phân tích DCF Monte Carlo để định giá cổ phiếu
""")

# Lấy danh sách mã cổ phiếu HOSE
listing = Listing()
symbols = listing.symbols_by_group('HOSE')

# Tạo sidebar cho nhập liệu
with st.sidebar:
    st.header("Thiết lập")
    
    # Nhập mã cổ phiếu
    symbol_input = st.text_input("Nhập mã cổ phiếu").strip()
    symbol_valid = next((s for s in symbols if s.lower() == symbol_input.lower()), None)

    if symbol_input and not symbol_valid:
        st.error("⚠️ Mã cổ phiếu không hợp lệ. Vui lòng nhập mã thuộc sàn HOSE.")
        st.stop()

    symbol = symbol_valid if symbol_valid else symbols[0]

    # Chọn loại phân tích
    analysis_type = st.radio(
        "Loại phân tích",
        ["Dữ liệu tài chính", "Giá lịch sử", "Định giá DCF Monte Carlo"],
        index=0
    )
    
    if analysis_type == "Dữ liệu tài chính":
        period = st.selectbox("Chu kỳ báo cáo", options=['year', 'quarter'], index=0)
        lang = st.selectbox("Ngôn ngữ", options=['vi', 'en'], index=0)
    elif analysis_type == "Giá lịch sử":
        # Thiết lập ngày cho dữ liệu giá
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Từ ngày",
                value=datetime.now() - timedelta(days=30),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "Đến ngày",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        interval = st.selectbox(
            "Khung thời gian",
            options=['1m', '5m', '15m', '30m', '1H', '1D', '1W', '1M'],
            index=5
        )
    else:
        # Thiết lập tham số cho DCF Monte Carlo
        st.subheader("Tham số định giá")
        
        # Tạo 2 cột cho các tham số
        col1, col2 = st.columns(2)
        
        with col1:
            wacc = st.number_input("WACC (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1) / 100
            num_simulations = st.number_input("Số lần mô phỏng", min_value=1000, max_value=50000, value=10000, step=1000)
        
        with col2:
            terminal_growth = st.number_input("Tỷ lệ tăng trưởng dài hạn (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100
            forecast_years = st.number_input("Số năm dự phóng", min_value=5, max_value=10, value=6, step=1)
        
        st.markdown("---")
        st.markdown("**Lưu ý:** Số năm dự phóng càng dài, độ chính xác của mô hình càng thấp do độ bất định tăng theo thời gian.")
    
   # Hàm xử lý dữ liệu
def clean_data(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    if df.columns.duplicated().any():
        columns = df.columns
        seen = {}
        new_columns = []
        for col in columns:
            if col in seen:
                seen[col] += 1
                new_col = f"{col}_{seen[col]}"
            else:
                seen[col] = 1
                new_col = col
            new_columns.append(new_col)
        df.columns = new_columns
    
    return df

# Class DCF Monte Carlo
class StockDCFMonteCarlo:
    def __init__(self, kqkd_data, lctt_data, cdkt_data, cstc_data, wacc, terminal_growth, num_simulations=10000, forecast_years=6):
        # Lọc dữ liệu từ năm 2020
        self.kqkd = kqkd_data[kqkd_data['Năm'] >= 2020].sort_values('Năm').copy()
        self.lctt = lctt_data[lctt_data['Năm'] >= 2020].sort_values('Năm').copy()
        self.cdkt = cdkt_data[cdkt_data['Năm'] >= 2020].sort_values('Năm').copy()
        self.cstc = cstc_data[cstc_data['Năm'] >= 2020].sort_values('Năm').copy()
        
        self.wacc = wacc
        self.terminal_growth = terminal_growth
        self.num_simulations = num_simulations
        self.forecast_years = forecast_years
        
        self.calculate_historical_ratios()
    
    def calculate_historical_ratios(self):
        """Tính toán tỷ lệ lịch sử và thống kê của chúng"""
        # Tỷ lệ tăng trưởng doanh thu thuần
        self.revenue_growth = []
        for i in range(1, len(self.kqkd)):
            growth = (self.kqkd['Doanh thu thuần'].iloc[i] - self.kqkd['Doanh thu thuần'].iloc[i-1]) / self.kqkd['Doanh thu thuần'].iloc[i-1]
            self.revenue_growth.append(growth)
        
        self.revenue_growth = pd.Series(self.revenue_growth)
        self.revenue_growth_mean = self.revenue_growth.mean()
        self.revenue_growth_std = self.revenue_growth.std()
        self.revenue_growth_min = self.revenue_growth.min()
        self.revenue_growth_max = self.revenue_growth.max()
        
        # Biên lợi nhuận hoạt động
        self.operating_margin = self.kqkd['Lãi/Lỗ từ hoạt động kinh doanh'] / self.kqkd['Doanh thu thuần']
        self.operating_margin_mean = self.operating_margin.mean()
        self.operating_margin_std = self.operating_margin.std()
        self.operating_margin_min = self.operating_margin.min()
        self.operating_margin_max = self.operating_margin.max()
        
        # Tỷ lệ thuế
        tax_current = self.kqkd['Chi phí thuế TNDN hiện hành']
        tax_deferred = self.kqkd['Chi phí thuế TNDN hoãn lại']
        total_tax = tax_current + tax_deferred
        self.tax_rate = abs(total_tax) / abs(self.kqkd['Lãi/Lỗ từ hoạt động kinh doanh'])
        self.tax_rate_mean = self.tax_rate.mean()
        self.tax_rate_std = self.tax_rate.std()
        self.tax_rate_min = self.tax_rate.min()
        self.tax_rate_max = self.tax_rate.max()
        
        # Tỷ lệ khấu hao
        self.depreciation_rate = abs(self.lctt['Khấu hao TSCĐ']) / self.kqkd['Doanh thu thuần']
        self.depreciation_rate_mean = self.depreciation_rate.mean()
        self.depreciation_rate_std = self.depreciation_rate.std()
        self.depreciation_rate_min = self.depreciation_rate.min()
        self.depreciation_rate_max = self.depreciation_rate.max()
        
        # Tỷ lệ Capex
        self.capex_rate_mean = 0.09
        self.capex_rate_std = 0.01
        self.capex_rate_min = 0.03
        self.capex_rate_max = 0.15
        
        # Tỷ lệ NWC
        self.nwc = self.cdkt['TÀI SẢN NGẮN HẠN (đồng)'] - self.cdkt['Nợ ngắn hạn (đồng)']
        self.nwc_change = self.nwc.diff()
        self.nwc_rate = self.nwc_change / self.kqkd['Doanh thu thuần']
        self.nwc_rate = self.nwc_rate.dropna()
        
        self.nwc_rate_mean = self.nwc_rate.mean()
        self.nwc_rate_std = self.nwc_rate.std()
        self.nwc_rate_min = self.nwc_rate.min()
        self.nwc_rate_max = self.nwc_rate.max()

    def generate_forecast(self):
        """Tạo dự báo cho một mô phỏng"""
        latest_revenue = self.kqkd['Doanh thu thuần'].iloc[-1]
        latest_nwc = self.nwc.iloc[-1]
        
        revenue_growth = np.random.normal(self.revenue_growth_mean, self.revenue_growth_std, self.forecast_years)
        operating_margins = np.random.normal(self.operating_margin_mean, self.operating_margin_std, self.forecast_years)
        tax_rates = np.random.normal(self.tax_rate_mean, self.tax_rate_std, self.forecast_years)
        depreciation_rates = np.random.normal(self.depreciation_rate_mean, self.depreciation_rate_std, self.forecast_years)
        capex_rates = np.random.normal(self.capex_rate_mean, self.capex_rate_std, self.forecast_years)
        nwc_rates = np.random.normal(self.nwc_rate_mean, self.nwc_rate_std, self.forecast_years)
        
        revenue_growth = np.clip(revenue_growth, self.revenue_growth_min, self.revenue_growth_max)
        operating_margins = np.clip(operating_margins, self.operating_margin_min, self.operating_margin_max)
        tax_rates = np.clip(tax_rates, self.tax_rate_min, self.tax_rate_max)
        depreciation_rates = np.clip(depreciation_rates, self.depreciation_rate_min, self.depreciation_rate_max)
        capex_rates = np.clip(capex_rates, self.capex_rate_min, self.capex_rate_max)
        nwc_rates = np.clip(nwc_rates, self.nwc_rate_min, self.nwc_rate_max)
        
        revenues = [latest_revenue]
        for growth in revenue_growth:
            revenues.append(revenues[-1] * (1 + growth))
        
        ebit = [rev * margin for rev, margin in zip(revenues[1:], operating_margins)]
        total_tax = [eb * tax for eb, tax in zip(ebit, tax_rates)]
        ebitat = [eb - abs(tax) for eb, tax in zip(ebit, total_tax)]
        
        depreciation = [rev * dep_rate for rev, dep_rate in zip(revenues[1:], depreciation_rates)]
        capex = [rev * cap_rate for rev, cap_rate in zip(revenues[1:], capex_rates)]
        nwc_changes = [rev * rate for rev, rate in zip(revenues[1:], nwc_rates)]
        
        fcf = [eb + dep - cap - nwc for eb, dep, cap, nwc in zip(ebitat, depreciation, capex, nwc_changes)]
        pv_fcf = [cf / (1 + self.wacc)**(i+1) for i, cf in enumerate(fcf)]
        
        terminal_value = fcf[-1] * (1 + self.terminal_growth) / (self.wacc - self.terminal_growth)
        pv_terminal = terminal_value / (1 + self.wacc)**self.forecast_years
        
        ev = sum(pv_fcf) + pv_terminal
        
        debt = (self.cdkt['Vay và nợ thuê tài chính dài hạn (đồng)'].iloc[-1] + 
                self.cdkt['Vay và nợ thuê tài chính ngắn hạn (đồng)'].iloc[-1])
        cash = self.cdkt['Tiền và tương đương tiền (đồng)'].iloc[-1]
        equity_value = ev - debt + cash
        
        shares = self.cstc['Số CP lưu hành (Triệu CP)'].iloc[-1]
        stock_price = max(0, equity_value / shares)
        
        return {
            'years': list(range(2025, 2025 + self.forecast_years)),
            'revenues': revenues[1:],
            'revenue_growth': revenue_growth,
            'operating_margins': operating_margins,
            'tax_rates': tax_rates,
            'depreciation_rates': depreciation_rates,
            'capex_rates': capex_rates,
            'nwc_rates': nwc_rates,
            'ebit': ebit,
            'total_tax': total_tax,
            'ebitat': ebitat,
            'depreciation': depreciation,
            'capex': capex,
            'nwc_changes': nwc_changes,
            'fcf': fcf,
            'pv_fcf': pv_fcf,
            'terminal_value': terminal_value,
            'pv_terminal': pv_terminal,
            'enterprise_value': ev,
            'equity_value': equity_value,
            'stock_price': stock_price
        }
    
    def run_simulation(self):
        """Chạy mô phỏng Monte Carlo"""
        stock_prices = []
        forecast_details_list = []
        for _ in range(self.num_simulations):
            forecast_details = self.generate_forecast()
            if forecast_details['stock_price'] > 0:
                stock_prices.append(forecast_details['stock_price'])
                forecast_details_list.append(forecast_details)
        return stock_prices, forecast_details_list
    
    def create_historical_ratios_df(self):
        """Tạo DataFrame cho tỷ lệ lịch sử"""
        historical_data = {
            'Tỷ Lệ': [
                'Tăng Trưởng Doanh Thu Thuần',
                'Biên Lợi Nhuận Hoạt Động',
                'Tỷ Lệ Thuế',
                'Tỷ Lệ Khấu Hao',
                'Tỷ Lệ Capex',
                'Tỷ Lệ NWC'
            ],
            'Trung Bình': [
                f"{self.revenue_growth_mean:.2%}",
                f"{self.operating_margin_mean:.2%}",
                f"{self.tax_rate_mean:.2%}",
                f"{self.depreciation_rate_mean:.2%}",
                f"{self.capex_rate_mean:.2%}",
                f"{self.nwc_rate_mean:.2%}"
            ],
            'Tối Thiểu': [
                f"{self.revenue_growth_min:.2%}",
                f"{self.operating_margin_min:.2%}",
                f"{self.tax_rate_min:.2%}",
                f"{self.depreciation_rate_min:.2%}",
                f"{self.capex_rate_min:.2%}",
                f"{self.nwc_rate_min:.2%}"
            ],
            'Tối Đa': [
                f"{self.revenue_growth_max:.2%}",
                f"{self.operating_margin_max:.2%}",
                f"{self.tax_rate_max:.2%}",
                f"{self.depreciation_rate_max:.2%}",
                f"{self.capex_rate_max:.2%}",
                f"{self.nwc_rate_max:.2%}"
            ]
        }
        return pd.DataFrame(historical_data)
    
    def create_valuation_statistics_df(self, stock_prices):
        """Tạo DataFrame cho thống kê định giá"""
        mean_price = np.mean(stock_prices)
        ci_lower = np.percentile(stock_prices, 2.5)
        ci_upper = np.percentile(stock_prices, 97.5)
        
        stats_data = {
            'Giá Cổ Phiếu Trung Bình (VND)': [f"{mean_price:,.2f}"],
            'Khoảng Tin Cậy 95% Thấp (VND)': [f"{ci_lower:,.2f}"],
            'Khoảng Tin Cậy 95% Cao (VND)': [f"{ci_upper:,.2f}"],
            'Độ Lệch Chuẩn (VND)': [f"{np.std(stock_prices):,.2f}"]
        }
        return pd.DataFrame(stats_data)
    
    def create_forecast_df(self, forecast):
        """Tạo DataFrame cho dự báo chi tiết"""
        forecast_data = []
        for i in range(self.forecast_years):
            year_data = {
                'Năm': forecast['years'][i],
                'Tăng Trưởng Doanh Thu Thuần': f"{forecast['revenue_growth'][i]:.2%}",
                'Doanh Thu (VND)': f"{forecast['revenues'][i]:,.2f}",
                'Biên Lợi Nhuận Hoạt Động': f"{forecast['operating_margins'][i]:.2%}",
                'EBIT (VND)': f"{forecast['ebit'][i]:,.2f}",
                'Tỷ Lệ Thuế': f"{forecast['tax_rates'][i]:.2%}",
                'Thuế Tổng Cộng (VND)': f"{forecast['total_tax'][i]:,.2f}",
                'EBIAT (VND)': f"{forecast['ebitat'][i]:,.2f}",
                'Tỷ Lệ Khấu Hao': f"{forecast['depreciation_rates'][i]:.2%}",
                'Khấu Hao (VND)': f"{forecast['depreciation'][i]:,.2f}",
                'Tỷ Lệ Capex': f"{forecast['capex_rates'][i]:.2%}",
                'Capex (VND)': f"{forecast['capex'][i]:,.2f}",
                'Tỷ Lệ NWC': f"{forecast['nwc_rates'][i]:.2%}",
                'Sự Thay Đổi NWC (VND)': f"{forecast['nwc_changes'][i]:,.2f}",
                'Dòng Tiền Tự Do (VND)': f"{forecast['fcf'][i]:,.2f}",
                'Giá Trị Hiện Tại (VND)': f"{forecast['pv_fcf'][i]:,.2f}"
            }
            forecast_data.append(year_data)
        return pd.DataFrame(forecast_data)
    
    def create_final_values_df(self, forecast):
        """Tạo DataFrame cho giá trị cuối cùng"""
        final_data = {
            'Giá Trị Cuối Cùng (VND)': [f"{forecast['terminal_value']:,.2f}"],
            'Giá Trị Hiện Tại Giá Trị Cuối Cùng (VND)': [f"{forecast['pv_terminal']:,.2f}"],
            'Giá Trị Doanh Nghiệp (VND)': [f"{forecast['enterprise_value']:,.2f}"],
            'Giá Trị Vốn Chủ Sở Hữu (VND)': [f"{forecast['equity_value']:,.2f}"]
        }
        return pd.DataFrame(final_data)

# Tải dữ liệu và hiển thị
try:
    if analysis_type == "Giá lịch sử":
        # Lấy dữ liệu giá từ TCBS
        stock_data = Vnstock().stock(symbol=symbol, source='TCBS')
    else:
        # Lấy dữ liệu tài chính từ VCI
        stock_data = Vnstock().stock(symbol=symbol, source='VCI')
    
    if analysis_type == "Dữ liệu tài chính":
        # Tạo tabs cho các loại báo cáo
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Chỉ Số Tài Chính", 
            "📊 Kết Quả Kinh Doanh", 
            "🏦 Cân Đối Kế Toán",
            "💰 Lưu Chuyển Tiền Tệ"
        ])
        
        with tab1:
            st.subheader(f"Chỉ số tài chính - {symbol}")
            cstc = stock_data.finance.ratio(period=period, lang=lang)
            cstc = clean_data(cstc)
            st.dataframe(cstc, use_container_width=True)
            
            st.download_button(
                label="Tải xuống CSTC (CSV)",
                data=cstc.to_csv(index=False).encode('utf-8-sig'),
                file_name=f'CSTC_{symbol}_{period}.csv',
                mime='text/csv'
            )
        
        with tab2:
            st.subheader(f"Kết quả kinh doanh - {symbol}")
            kqkd = stock_data.finance.income_statement(period=period, lang=lang)
            kqkd = clean_data(kqkd)
            st.dataframe(kqkd, use_container_width=True)
            
            st.download_button(
                label="Tải xuống KQKD (CSV)",
                data=kqkd.to_csv(index=False).encode('utf-8-sig'),
                file_name=f'KQKD_{symbol}_{period}.csv',
                mime='text/csv'
            )
        
        with tab3:
            st.subheader(f"Cân đối kế toán - {symbol}")
            cdkt = stock_data.finance.balance_sheet(period=period, lang=lang)
            cdkt = clean_data(cdkt)
            st.dataframe(cdkt, use_container_width=True)
            
            st.download_button(
                label="Tải xuống CDKT (CSV)",
                data=cdkt.to_csv(index=False).encode('utf-8-sig'),
                file_name=f'CDKT_{symbol}_{period}.csv',
                mime='text/csv'
            )

        with tab4:
            st.subheader(f"Lưu Chuyển Tiền Tệ - {symbol}")
            lctt = stock_data.finance.cash_flow(period=period, lang=lang)
            lctt = clean_data(lctt)
            st.dataframe(lctt, use_container_width=True)
            
            st.download_button(
                label="Tải xuống LCTT (CSV)",
                data=lctt.to_csv(index=False).encode('utf-8-sig'),
                file_name=f'LCTT_{symbol}_{period}.csv',
                mime='text/csv'
            )
    
    elif analysis_type == "Giá lịch sử":
        st.subheader(f"Giá lịch sử - {symbol}")
        
        # Chuyển đổi ngày thành chuỗi
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Lấy dữ liệu giá
        price_data = stock_data.quote.history(
            start=start_str,
            end=end_str,
            interval=interval
        )
        
        # Tạo tabs cho các loại biểu đồ
        chart_tab1, chart_tab2, chart_tab3 = st.tabs([
            "📈 Biểu Đồ Giá",
            "📊 Chỉ Báo Kỹ Thuật",
            "📉 Phân Tích Xu Hướng"
        ])
        
        # Chọn các chỉ báo kỹ thuật
        st.sidebar.markdown("### Chỉ Báo Kỹ Thuật")
        
        # Tùy chọn hiển thị giá
        price_display = st.sidebar.radio(
            "Hiển thị giá", 
            ["Biểu đồ nến", "Giá đóng cửa"],
            index=0
        )
        
        # Tùy chọn cho biểu đồ giá
        st.sidebar.markdown("#### Biểu Đồ Giá")
        show_ma = st.sidebar.checkbox("Đường MA", value=True)
        if show_ma:
            ma20 = st.sidebar.checkbox("MA20", value=True)
            ma50 = st.sidebar.checkbox("MA50", value=True)
            ma100 = st.sidebar.checkbox("MA100", value=True)
        
        show_bb = st.sidebar.checkbox("Bollinger Bands", value=True)
        
        # Tùy chọn cho biểu đồ MACD
        st.sidebar.markdown("#### Biểu Đồ MACD")
        show_macd = st.sidebar.checkbox("MACD", value=True)
        if show_macd:
            show_macd_line = st.sidebar.checkbox("MACD Line", value=True)
            show_signal_line = st.sidebar.checkbox("Signal Line", value=True)
            show_histogram = st.sidebar.checkbox("MACD Histogram", value=True)
        
        # Tùy chọn cho biểu đồ Stochastic
        st.sidebar.markdown("#### Biểu Đồ Stochastic")
        show_stoch = st.sidebar.checkbox("Stochastic", value=True)
        if show_stoch:
            show_stoch_k = st.sidebar.checkbox("Stoch %K", value=True)
            show_stoch_d = st.sidebar.checkbox("Stoch %D", value=True)
        
        # Tính toán các chỉ báo
        if not price_data.empty and 'close' in price_data.columns:
            with chart_tab1:
                # Tạo figure cho biểu đồ giá
                fig_price = go.Figure()
                
                # Vẽ giá theo tùy chọn
                if price_display == "Biểu đồ nến":
                    fig_price.add_trace(go.Candlestick(
                        x=price_data['time'],
                        open=price_data['open'],
                        high=price_data['high'],
                        low=price_data['low'],
                        close=price_data['close'],
                        name='Giá',
                        increasing_line_color='#26a69a',
                        decreasing_line_color='#ef5350',
                        increasing_fillcolor='#26a69a',
                        decreasing_fillcolor='#ef5350'
                    ))
                else:
                    fig_price.add_trace(go.Scatter(
                        x=price_data['time'],
                        y=price_data['close'],
                        name='Giá đóng cửa',
                        line=dict(color='#2196F3', width=2)
                    ))
                
                # Vẽ các đường MA
                if show_ma:
                    if ma20:
                        ma20_data = ta.trend.sma_indicator(price_data['close'], window=20)
                        fig_price.add_trace(go.Scatter(
                            x=price_data['time'],
                            y=ma20_data,
                            name='MA20',
                            line=dict(color='#FF9800', width=1)
                        ))
                    if ma50:
                        ma50_data = ta.trend.sma_indicator(price_data['close'], window=50)
                        fig_price.add_trace(go.Scatter(
                            x=price_data['time'],
                            y=ma50_data,
                            name='MA50',
                            line=dict(color='#4CAF50', width=1)
                        ))
                    if ma100:
                        ma100_data = ta.trend.sma_indicator(price_data['close'], window=100)
                        fig_price.add_trace(go.Scatter(
                            x=price_data['time'],
                            y=ma100_data,
                            name='MA100',
                            line=dict(color='#9C27B0', width=1)
                        ))
                
                # Vẽ Bollinger Bands
                if show_bb:
                    bb = ta.volatility.BollingerBands(price_data['close'])
                    fig_price.add_trace(go.Scatter(
                        x=price_data['time'],
                        y=bb.bollinger_hband(),
                        name='BB Upper',
                        line=dict(color='#90A4AE', dash='dash', width=1)
                    ))
                    fig_price.add_trace(go.Scatter(
                        x=price_data['time'],
                        y=bb.bollinger_lband(),
                        name='BB Lower',
                        line=dict(color='#90A4AE', dash='dash', width=1),
                        fill='tonexty'
                    ))
                    fig_price.add_trace(go.Scatter(
                        x=price_data['time'],
                        y=bb.bollinger_mavg(),
                        name='BB Middle',
                        line=dict(color='#90A4AE', width=1)
                    ))
                
                # Cập nhật layout cho biểu đồ giá
                fig_price.update_layout(
                    title=f'Biểu Đồ Giá {symbol}',
                    yaxis_title='Giá (VND)',
                    xaxis_rangeslider_visible=False,
                    height=600,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    template='plotly_white'
                )
                
                # Thêm mốc thời gian
                fig_price.update_xaxes(
                    rangeslider_visible=False,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(step="all", label="Tất cả")
                        ])
                    )
                )
                
                st.plotly_chart(fig_price, use_container_width=True)
            
            with chart_tab2:
                # Tạo figure cho biểu đồ MACD
                fig_macd = go.Figure()
                
                if show_macd:
                    macd = ta.trend.MACD(price_data['close'])
                    if show_macd_line:
                        fig_macd.add_trace(go.Scatter(
                            x=price_data['time'],
                            y=macd.macd(),
                            name='MACD',
                            line=dict(color='#2196F3', width=2)
                        ))
                    if show_signal_line:
                        fig_macd.add_trace(go.Scatter(
                            x=price_data['time'],
                            y=macd.macd_signal(),
                            name='Signal',
                            line=dict(color='#FF9800', width=2)
                        ))
                    if show_histogram:
                        fig_macd.add_trace(go.Bar(
                            x=price_data['time'],
                            y=macd.macd_diff(),
                            name='MACD Histogram',
                            marker_color='#90A4AE'
                        ))
                
                # Cập nhật layout cho biểu đồ MACD
                fig_macd.update_layout(
                    title='MACD',
                    yaxis_title='MACD',
                    xaxis_rangeslider_visible=False,
                    height=400,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_macd, use_container_width=True)
            
            with chart_tab3:
                # Tạo figure cho biểu đồ Stochastic
                fig_stoch = go.Figure()
                
                if show_stoch:
                    stoch = ta.momentum.StochasticOscillator(price_data['high'], price_data['low'], price_data['close'])
                    if show_stoch_k:
                        fig_stoch.add_trace(go.Scatter(
                            x=price_data['time'],
                            y=stoch.stoch(),
                            name='Stoch %K',
                            line=dict(color='#2196F3', width=2)
                        ))
                    if show_stoch_d:
                        fig_stoch.add_trace(go.Scatter(
                            x=price_data['time'],
                            y=stoch.stoch_signal(),
                            name='Stoch %D',
                            line=dict(color='#FF9800', width=2)
                        ))
                    fig_stoch.add_hline(y=80, line_dash="dash", line_color="#ef5350")
                    fig_stoch.add_hline(y=20, line_dash="dash", line_color="#26a69a")
                
                # Cập nhật layout cho biểu đồ Stochastic
                fig_stoch.update_layout(
                    title='Stochastic Oscillator',
                    yaxis_title='Stochastic',
                    xaxis_rangeslider_visible=False,
                    height=400,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_stoch, use_container_width=True)
        
        # Hiển thị dữ liệu
        st.dataframe(price_data, use_container_width=True)
        
        # Nút tải xuống
        st.download_button(
            label="Tải xuống dữ liệu giá (CSV)",
            data=price_data.to_csv(index=False).encode('utf-8-sig'),
            file_name=f'Price_{symbol}_{start_str}_to_{end_str}_{interval}.csv',
            mime='text/csv'
        )
    
    else:  # Phân tích DCF Monte Carlo
        st.subheader(f"Định giá DCF Monte Carlo - {symbol}")
        
        # Lấy dữ liệu tài chính
        kqkd = clean_data(stock_data.finance.income_statement(period='year', lang='vi'))
        lctt = clean_data(stock_data.finance.cash_flow(period='year', lang='vi'))
        cdkt = clean_data(stock_data.finance.balance_sheet(period='year', lang='vi'))
        cstc = clean_data(stock_data.finance.ratio(period='year', lang='vi'))
        
        # Chuyển đổi các cột số từ string sang numeric
        for df in [kqkd, lctt, cdkt]:
            for col in df.columns:
                if col not in ['Thời gian', 'Thời gian báo cáo', 'Năm']:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Thêm cột Năm vào các DataFrame
        for df in [kqkd, lctt, cdkt, cstc]:
            if 'Thời gian' in df.columns:
                df['Năm'] = pd.to_datetime(df['Thời gian']).dt.year
            elif 'Thời gian báo cáo' in df.columns:
                df['Năm'] = pd.to_datetime(df['Thời gian báo cáo']).dt.year
            elif 'Năm' in df.columns:
                df['Năm'] = df['Năm'].astype(int)
        
        # Khởi tạo và chạy mô hình
        model = StockDCFMonteCarlo(
            kqkd_data=kqkd,
            lctt_data=lctt,
            cdkt_data=cdkt,
            cstc_data=cstc,
            wacc=wacc,
            terminal_growth=terminal_growth,
            num_simulations=num_simulations,
            forecast_years=forecast_years
        )
        
        # Chạy mô phỏng và hiển thị kết quả
        stock_prices, forecast_details_list = model.run_simulation()
        
        # Hiển thị kết quả định giá ngay từ đầu
        st.markdown("### Kết Quả Định Giá")
        
        # Tính toán thống kê từ kết quả mô phỏng
        mean_price = np.mean(stock_prices)
        
        # Hiển thị giá trị định giá
        st.metric(
            label="Giá Trị Được Định Giá",
            value=f"{mean_price:,.0f} VND"
        )
        
        # Hiển thị các phần phân tích chi tiết
        st.markdown("### Phân Tích Tỷ Lệ Lịch Sử (2020-2024)")
        st.dataframe(model.create_historical_ratios_df(), use_container_width=True)
        
        st.markdown(f"### Dự Báo {forecast_years} Năm Chi Tiết (Mô Phỏng Đầu Tiên)")
        st.dataframe(model.create_forecast_df(forecast_details_list[0]), use_container_width=True)
        
        st.markdown("### Giá Trị Cuối Cùng")
        st.dataframe(model.create_final_values_df(forecast_details_list[0]), use_container_width=True)
        
        # Thêm phần biểu đồ dự báo
        st.markdown("### Biểu Đồ Dự Báo Chi Tiết")
        
        # Tạo danh sách các chỉ số có thể xem
        forecast_metrics = {
            'Doanh Thu': {
                'value': forecast_details_list[0]['revenues'],
                'growth': forecast_details_list[0]['revenue_growth']
            },
            'EBIT': {
                'value': forecast_details_list[0]['ebit'],
                'growth': forecast_details_list[0]['operating_margins']
            },
            'Thuế': {
                'value': forecast_details_list[0]['total_tax'],
                'growth': forecast_details_list[0]['tax_rates']
            },
            'Khấu Hao': {
                'value': forecast_details_list[0]['depreciation'],
                'growth': forecast_details_list[0]['depreciation_rates']
            },
            'Capex': {
                'value': forecast_details_list[0]['capex'],
                'growth': forecast_details_list[0]['capex_rates']
            },
            'Thay Đổi NWC': {
                'value': forecast_details_list[0]['nwc_changes'],
                'growth': forecast_details_list[0]['nwc_rates']
            },
            'Dòng Tiền Tự Do': {
                'value': forecast_details_list[0]['fcf'],
                'growth': None
            }
        }
        
        # Tạo selectbox để chọn chỉ số
        selected_metric = st.selectbox(
            "Chọn chỉ số để xem biểu đồ",
            options=list(forecast_metrics.keys())
        )
        
        # Vẽ biểu đồ cho chỉ số được chọn
        if selected_metric:
            fig = go.Figure()
            
            # Thêm biểu đồ cột cho giá trị
            fig.add_trace(go.Bar(
                x=forecast_details_list[0]['years'],
                y=forecast_metrics[selected_metric]['value'],
                name=f"{selected_metric}",
                marker_color='#2196F3'
            ))
            
            # Thêm biểu đồ đường cho tỷ lệ tăng trưởng (nếu có)
            if forecast_metrics[selected_metric]['growth'] is not None:
                fig.add_trace(go.Scatter(
                    x=forecast_details_list[0]['years'],
                    y=[x * 100 for x in forecast_metrics[selected_metric]['growth']],  # Chuyển đổi sang phần trăm
                    name=f"Tỷ lệ {selected_metric}",
                    line=dict(color='#FF9800', width=2),
                    yaxis='y2'
                ))
            
            # Cập nhật layout
            fig.update_layout(
                title=dict(
                    text=f'Dự Báo {selected_metric} ({forecast_years} Năm)',
                    font=dict(size=16)
                ),
                yaxis=dict(
                    title=dict(
                        text=f"{selected_metric} (VND)",
                        font=dict(color='#2196F3')
                    ),
                    tickfont=dict(color='#2196F3')
                ),
                yaxis2=dict(
                    title=dict(
                        text="Tỷ lệ (%)",
                        font=dict(color='#FF9800')
                    ),
                    tickfont=dict(color='#FF9800'),
                    overlaying='y',
                    side='right'
                ) if forecast_metrics[selected_metric]['growth'] is not None else None,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=500,
                template='plotly_white'
            )
            
            # Hiển thị biểu đồ
            st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị biểu đồ phân phối và box plot ở cuối
        st.markdown("### Phân Phối Giá Trị Định Giá")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Biểu đồ histogram với đường KDE và đường trung bình
        sns.histplot(stock_prices, kde=True, ax=ax1)
        ax1.axvline(np.mean(stock_prices), color='red', linestyle='--', label=f'Mean: {np.mean(stock_prices):,.2f}')
        ax1.axvline(np.median(stock_prices), color='green', linestyle='--', label=f'Median: {np.median(stock_prices):,.2f}')
        ax1.set_title('Phân Phối Giá Cổ Phiếu')
        ax1.set_xlabel('Giá Cổ Phiếu (VND)')
        ax1.set_ylabel('Tần Suất')
        ax1.legend()
        
        # Biểu đồ box plot
        sns.boxplot(y=stock_prices, ax=ax2)
        ax2.set_title('Box Plot Giá Cổ Phiếu')
        ax2.set_ylabel('Giá Cổ Phiếu (VND)')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Hiển thị bảng thống kê chi tiết
        st.markdown("### Thống Kê Chi Tiết")
        stats_df = pd.DataFrame({
            'Chỉ số': [
                'Giá trị trung bình',
                'Giá trị trung vị',
                'Độ lệch chuẩn',
                'Giá trị thấp nhất',
                'Giá trị cao nhất',
                'Phân vị 25%',
                'Phân vị 75%',
                'Độ xiên',
                'Độ nhọn'
            ],
            'Giá trị': [
                f"{mean_price:,.0f} VND",
                f"{np.median(stock_prices):,.0f} VND",
                f"{np.std(stock_prices):,.0f} VND",
                f"{np.min(stock_prices):,.0f} VND",
                f"{np.max(stock_prices):,.0f} VND",
                f"{np.percentile(stock_prices, 25):,.0f} VND",
                f"{np.percentile(stock_prices, 75):,.0f} VND",
                f"{stats.skew(stock_prices):.2f}",
                f"{stats.kurtosis(stock_prices):.2f}"
            ]
        })
        st.dataframe(stats_df, hide_index=True)

except Exception as e:
    st.error(f"⚠️ Có lỗi xảy ra: {str(e)}")
    st.info("Vui lòng kiểm tra lại mã cổ phiếu hoặc thử lại sau.") 