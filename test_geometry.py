#!/usr/bin/env python3
"""
測試羽球場幾何模型的正確性
確保我們生成的交點數量和位置合理
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

class BadmintonCourtGeometry:
    """羽球場標準尺寸和幾何約束"""
    
    # 羽球場標準尺寸 (米)
    COURT_LENGTH = 13.4
    COURT_WIDTH = 6.1
    SERVICE_LINE_DISTANCE = 1.98  # 從網線到發球線  
    DOUBLES_SIDELINE_DISTANCE = 0.46  # 單雙打邊線距離
    
    @classmethod
    def generate_standard_court_grid(cls):
        """
        根據標準羽球場規格生成 6x5 = 30個交點矩陣
        
        矩陣佈局 (從上到下，從左到右):
        00:L-corner    01:T-junction  02:T-junction  03:T-junction  04:L-corner
        10:T-junction  11:Cross       12:Cross       13:Cross       14:T-junction  
        20:T-junction  21:T-junction  22:T-junction  23:T-junction  24:T-junction
        30:T-junction  31:T-junction  32:T-junction  33:T-junction  34:T-junction
        40:T-junction  41:Cross       42:Cross       43:Cross       44:T-junction
        50:T-junction  51:T-junction  52:T-junction  53:T-junction  54:T-junction
        """
        
        # 定義6行5列的網格
        rows = 6  # Y方向
        cols = 5  # X方向
        
        # 根據羽球場標準尺寸計算正規化座標
        # X座標: 從左邊線到右邊線 (-0.5 到 0.5)
        x_positions = [-0.5, -0.25, 0.0, 0.25, 0.5]  # 5個X位置
        
        # Y座標: 從頂線到底線 (0.5 到 -0.6) 
        y_positions = [0.5, 0.295, 0.0, -0.295, -0.5, -0.6]  # 6個Y位置
        
        intersections = {}
        
        # 預定義每個位置的交點類型 (按您提供的矩陣)
        intersection_types = [
            [0, 1, 1, 1, 0],  # 第0行: L-corner, T-junction, T-junction, T-junction, L-corner
            [1, 2, 2, 2, 1],  # 第1行: T-junction, Cross, Cross, Cross, T-junction
            [1, 1, 1, 1, 1],  # 第2行: T-junction, T-junction, T-junction, T-junction, T-junction
            [1, 1, 1, 1, 1],  # 第3行: T-junction, T-junction, T-junction, T-junction, T-junction
            [1, 2, 2, 2, 1],  # 第4行: T-junction, Cross, Cross, Cross, T-junction
            [1, 1, 1, 1, 1],  # 第5行: T-junction, T-junction, T-junction, T-junction, T-junction
        ]
        
        intersection_id = 0
        for row in range(rows):
            for col in range(cols):
                x = x_positions[col]
                y = y_positions[row]
                intersection_type = intersection_types[row][col]
                
                intersections[f'point_{row}{col}'] = {
                    'coords': (x, y),
                    'type': intersection_type,
                    'id': intersection_id,
                    'grid_pos': (row, col)
                }
                intersection_id += 1
        
        return intersections
    
    @classmethod
    def generate_all_intersections(cls):
        """使用標準網格生成交點"""
        return cls.generate_standard_court_grid()

def test_geometry_model():
    """測試幾何模型是否正確生成交點"""
    geometry = BadmintonCourtGeometry()
    intersections = geometry.generate_all_intersections()
    
    print(f"Generated {len(intersections)} intersection points:")
    
    # 按類型統計
    type_counts = {0: 0, 1: 0, 2: 0}  # L角, T字, 十字
    type_names = {0: 'L角', 1: 'T字', 2: '十字'}
    
    print("\n詳細交點列表:")
    for name, info in sorted(intersections.items()):
        intersection_type = info['type']
        coords = info['coords']
        type_counts[intersection_type] += 1
        print(f"{name:30}: {coords[0]:6.3f}, {coords[1]:6.3f} -> {type_names[intersection_type]}")
    
    print(f"\n統計:")
    for type_id, count in type_counts.items():
        print(f"{type_names[type_id]}: {count}個")
    
    total_intersections = sum(type_counts.values())
    print(f"總計: {total_intersections}個交點")
    
    # 檢查是否符合預期的30個交點
    if total_intersections == 30:
        print("✅ 交點數量正確 (6x5 = 30個)")
    else:
        print(f"⚠️  交點數量不符合預期: {total_intersections}, 應為30個")
    
    return intersections

def visualize_court(intersections):
    """視覺化羽球場和交點"""
    plt.figure(figsize=(12, 8))
    
    # 繪制場地線條
    lines = BadmintonCourtGeometry.get_court_lines()
    
    # 繪制橫線
    for line_name, line_info in lines.items():
        if 'y' in line_info:  # 橫線
            y = line_info['y']
            x_range = line_info['x_range']
            plt.plot([x_range[0], x_range[1]], [y, y], 'k-', linewidth=2, alpha=0.7)
            plt.text((x_range[0] + x_range[1])/2, y + 0.02, line_name, 
                    ha='center', va='bottom', fontsize=8)
    
    # 繪制縱線
    for line_name, line_info in lines.items():
        if 'x' in line_info:  # 縱線
            x = line_info['x']
            y_range = line_info['y_range']
            plt.plot([x, x], [y_range[0], y_range[1]], 'k-', linewidth=2, alpha=0.7)
            plt.text(x + 0.02, (y_range[0] + y_range[1])/2, line_name, 
                    ha='left', va='center', fontsize=8, rotation=90)
    
    # 繪制交點
    colors = {0: 'red', 1: 'blue', 2: 'green'}  # L角: 紅, T字: 藍, 十字: 綠
    type_names = {0: 'L角', 1: 'T字', 2: '十字'}
    
    for name, info in intersections.items():
        coords = info['coords']
        intersection_type = info['type']
        plt.scatter(coords[0], coords[1], c=colors[intersection_type], 
                   s=50, alpha=0.8, label=type_names[intersection_type])
    
    # 去除重複的legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.xlim(-0.6, 0.6)
    plt.ylim(-0.6, 0.6)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.title('羽球場交點分佈圖')
    plt.xlabel('X 座標 (正規化)')
    plt.ylabel('Y 座標 (正規化)')
    
    plt.tight_layout()
    plt.savefig('badminton_court_intersections.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_with_actual_data():
    """與實際標註數據比較"""
    # 從標註文件中我們看到有48個點
    actual_labels = [
        (0, 0.313103, 0.530711), (1, 0.341037, 0.530622), (1, 0.498393, 0.530122),
        (1, 0.655874, 0.529622), (0, 0.683857, 0.529533), (1, 0.309391, 0.545739),
        # ... 這裡應該包含所有48個點
    ]
    
    print(f"\n實際標註數據:")
    print(f"總點數: 48個")
    print(f"我們的模型生成: {len(test_geometry_model())}個")
    
    print("\n建議:")
    print("1. 檢查是否遺漏了某些特殊交點")
    print("2. 確認羽球場規格是否完整")
    print("3. 可能需要添加更多細分線條")

if __name__ == '__main__':
    print("=== 羽球場幾何模型測試 ===")
    
    # 測試幾何模型
    intersections = test_geometry_model()
    
    # 視覺化
    print(f"\n正在生成視覺化圖表...")
    visualize_court(intersections)
    
    # 與實際數據比較
    compare_with_actual_data()
    
    print(f"\n測試完成！圖表已保存為 'badminton_court_intersections.png'")