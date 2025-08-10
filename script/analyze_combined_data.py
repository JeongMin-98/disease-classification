#!/usr/bin/env python3
"""
converted_final_samples.json에서 data/foot/combined 경로가 포함된 데이터만 필터링하여 클래스 분포를 분석하는 스크립트
"""

import json
from collections import Counter, defaultdict
import os
import random

def analyze_combined_data(json_file_path):
    """
    JSON 파일에서 data/foot/combined 경로가 포함된 데이터만 필터링하여 클래스 분포를 분석
    
    Args:
        json_file_path (str): JSON 파일 경로
    
    Returns:
        dict: 클래스별 분포 정보
    """
    
    # JSON 파일 로드
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 전체 데이터의 클래스별 분포
    all_class_counts = Counter(record['class'] for record in data['data'])
    
    # data/foot/combined 경로가 포함된 데이터만 필터링
    combined_data = []
    for record in data['data']:
        file_path = record.get('file_path', '')
        if 'data/foot/combined' in file_path:
            combined_data.append(record)
    
    # combined 데이터의 클래스별 카운트
    combined_class_counts = Counter(record['class'] for record in combined_data)
    
    # 결과 출력
    print("=" * 80)
    print("data/foot/combined 경로가 포함된 데이터 분석 결과")
    print("=" * 80)
    print(f"총 데이터 수: {len(combined_data)}")
    print(f"전체 데이터 대비 비율: {len(combined_data)/len(data['data'])*100:.2f}%")
    print()
    
    print("클래스별 분포 및 비율:")
    print("-" * 60)
    print(f"{'클래스':<15} {'전체':<8} {'combined':<10} {'비율(%)':<10} {'균등224개':<12} {'균등비율(%)':<12}")
    print("-" * 60)
    
    for class_name in sorted(all_class_counts.keys()):
        all_count = all_class_counts[class_name]
        combined_count = combined_class_counts.get(class_name, 0)
        combined_ratio = (combined_count / all_count * 100) if all_count > 0 else 0
        
        # 균등 샘플링 시 예상 개수 (224개 기준)
        total_combined = len(combined_data)
        if total_combined > 0:
            expected_count = int(224 * combined_count / total_combined)
            expected_ratio = (expected_count / all_count * 100) if all_count > 0 else 0
        else:
            expected_count = 0
            expected_ratio = 0
        
        print(f"{class_name:<15} {all_count:<8} {combined_count:<10} {combined_ratio:<10.1f} {expected_count:<12} {expected_ratio:<12.1f}")
    
    print()
    
    # 균등 샘플링 시뮬레이션
    print("균등 샘플링 시뮬레이션 (random seed 42):")
    print("-" * 50)
    
    # random seed 설정
    random.seed(42)
    
    # 균등 샘플링 (224개)
    if len(combined_data) >= 224:
        sampled_data = random.sample(combined_data, 224)
    else:
        sampled_data = combined_data  # 224개보다 적으면 전체 사용
    
    sampled_class_counts = Counter(record['class'] for record in sampled_data)
    
    print(f"샘플링된 데이터 수: {len(sampled_data)}")
    print()
    print(f"{'클래스':<15} {'샘플링된 수':<12} {'전체 대비 비율(%)':<15}")
    print("-" * 45)
    
    for class_name in sorted(sampled_class_counts.keys()):
        sampled_count = sampled_class_counts[class_name]
        all_count = all_class_counts[class_name]
        ratio = (sampled_count / all_count * 100) if all_count > 0 else 0
        print(f"{class_name:<15} {sampled_count:<12} {ratio:<15.1f}")
    
    print()
    
    # 상세 정보
    print("상세 정보:")
    print("-" * 30)
    
    # 각 클래스별 샘플 정보
    for class_name in sorted(combined_class_counts.keys()):
        class_samples = [record for record in combined_data if record['class'] == class_name]
        print(f"\n{class_name} 클래스 ({len(class_samples)}개):")
        
        # 처음 3개 샘플의 파일 경로 출력
        for i, sample in enumerate(class_samples[:3]):
            print(f"  {i+1}. {sample['file_path']}")
        
        if len(class_samples) > 3:
            print(f"  ... 외 {len(class_samples)-3}개 더")
    
    return {
        'total_count': len(combined_data),
        'class_distribution': dict(combined_class_counts),
        'all_class_distribution': dict(all_class_counts),
        'sampled_distribution': dict(sampled_class_counts),
        'samples': combined_data,
        'sampled_samples': sampled_data
    }

def save_filtered_data(analysis_result, output_file):
    """
    필터링된 데이터를 새로운 JSON 파일로 저장
    
    Args:
        analysis_result (dict): 분석 결과
        output_file (str): 출력 파일 경로
    """
    output_data = {
        'meta': {
            'total_records': analysis_result['total_count'],
            'class_distribution': analysis_result['class_distribution'],
            'all_class_distribution': analysis_result['all_class_distribution'],
            'sampled_distribution': analysis_result['sampled_distribution'],
            'filter_criteria': 'data/foot/combined 경로 포함'
        },
        'data': analysis_result['samples']
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n필터링된 데이터가 {output_file}에 저장되었습니다.")

def save_sampled_data(analysis_result, output_file):
    """
    균등 샘플링된 데이터를 새로운 JSON 파일로 저장
    
    Args:
        analysis_result (dict): 분석 결과
        output_file (str): 출력 파일 경로
    """
    output_data = {
        'meta': {
            'total_records': len(analysis_result['sampled_samples']),
            'class_distribution': analysis_result['sampled_distribution'],
            'sampling_method': '균등 샘플링 (random seed 42)',
            'original_total': analysis_result['total_count']
        },
        'data': analysis_result['sampled_samples']
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"균등 샘플링된 데이터가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    # 파일 경로 설정
    json_file = "data/json/converted_final_samples.json"
    output_file = "data/json/combined_samples.json"
    sampled_output_file = "data/json/combined_samples_224.json"
    
    # 파일 존재 확인
    if not os.path.exists(json_file):
        print(f"오류: {json_file} 파일을 찾을 수 없습니다.")
        exit(1)
    
    # 분석 실행
    try:
        result = analyze_combined_data(json_file)
        
        # 필터링된 데이터 저장
        save_filtered_data(result, output_file)
        
        # 균등 샘플링된 데이터 저장
        save_sampled_data(result, sampled_output_file)
        
    except Exception as e:
        print(f"오류 발생: {e}")
        exit(1) 