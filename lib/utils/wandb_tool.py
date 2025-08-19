#!/usr/bin/env python3
"""
wandb 관련 유틸리티 함수들
- 모델 파일 다운로드
- config 파일 다운로드
- 다운로드된 파일 관리 및 재사용
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("경고: wandb가 설치되지 않았습니다. pip install wandb로 설치해주세요.")

class WandbDownloader:
    """wandb에서 파일을 다운로드하고 관리하는 클래스"""
    
    def __init__(self, project_name: str = "jmkim/disease-classification", 
                 download_dir: str = "downloaded_files"):
        self.project_name = project_name
        self.download_dir = download_dir
        self.metadata_file = os.path.join(download_dir, "download_metadata.json")
        
        # 다운로드 디렉토리 생성
        os.makedirs(download_dir, exist_ok=True)
        
        # 메타데이터 로드
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """다운로드 메타데이터를 로드합니다."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"메타데이터 로드 실패: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """다운로드 메타데이터를 저장합니다."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"메타데이터 저장 실패: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """파일의 해시값을 계산합니다."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _is_file_valid(self, file_path: str, expected_hash: str = None) -> bool:
        """파일이 유효한지 확인합니다."""
        if not os.path.exists(file_path):
            return False
        
        if expected_hash:
            current_hash = self._get_file_hash(file_path)
            return current_hash == expected_hash
        
        return True
    
    def download_model(self, run_id: str, filename: str = "best_model.pth", 
                      force_download: bool = False) -> str:
        """모델 파일을 다운로드합니다."""
        if not WANDB_AVAILABLE:
            raise ImportError("wandb가 설치되지 않았습니다. pip install wandb로 설치해주세요.")
        
        # 메타데이터 키 생성
        metadata_key = f"model_{run_id}_{filename}"
        
        # 이미 다운로드된 파일이 있고 강제 다운로드가 아닌 경우
        if not force_download and metadata_key in self.metadata:
            cached_info = self.metadata[metadata_key]
            cached_path = cached_info.get('file_path', '')
            
            if self._is_file_valid(cached_path, cached_info.get('file_hash')):
                print(f"캐시된 모델 파일 사용: {cached_path}")
                return cached_path
        
        try:
            print(f"wandb에서 모델 다운로드 중... (run_id: {run_id})")
            
            # wandb API 초기화
            api = wandb.Api()
            
            # run 정보 가져오기
            run = api.run(f"{self.project_name}/{run_id}")
            print(f"Run 이름: {run.name}")
            print(f"Run 상태: {run.state}")
            
            # 파일 목록 확인
            files = run.files()
            print(f"사용 가능한 파일들:")
            for file in files:
                print(f"  - {file.name} ({file.size} bytes)")
            
            # target 파일 찾기
            target_file = None
            for file in files:
                if filename in file.name:
                    target_file = file
                    break
            
            if target_file is None:
                raise FileNotFoundError(f"'{filename}' 파일을 찾을 수 없습니다.")
            
            # 파일 다운로드
            output_path = os.path.join(self.download_dir, f"{run_id}_{filename}")
            print(f"다운로드 중: {target_file.name} -> {output_path}")
            
            target_file.download(output_path)
            
            # 메타데이터 업데이트
            file_hash = self._get_file_hash(output_path)
            self.metadata[metadata_key] = {
                'file_path': output_path,
                'file_hash': file_hash,
                'run_id': run_id,
                'filename': filename,
                'run_name': run.name,
                'download_time': str(Path(output_path).stat().st_mtime),
                'file_size': target_file.size
            }
            self._save_metadata()
            
            print(f"모델 다운로드 완료: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"wandb에서 모델 다운로드 실패: {e}")
            print(f"\nwandb 인증이 필요할 수 있습니다:")
            print(f"1. wandb login 명령어로 로그인")
            print(f"2. WANDB_API_KEY 환경변수 설정")
            print(f"3. 또는 ~/.netrc 파일에 API 키 설정")
            raise
    
    def download_config(self, run_id: str, force_download: bool = False) -> str:
        """config 파일을 다운로드합니다."""
        if not WANDB_AVAILABLE:
            raise ImportError("wandb가 설치되지 않았습니다. pip install wandb로 설치해주세요.")
        
        # 메타데이터 키 생성
        metadata_key = f"config_{run_id}"
        
        # 이미 다운로드된 파일이 있고 강제 다운로드가 아닌 경우
        if not force_download and metadata_key in self.metadata:
            cached_info = self.metadata[metadata_key]
            cached_path = cached_info.get('file_path', '')
            
            if self._is_file_valid(cached_path, cached_info.get('file_hash')):
                print(f"캐시된 config 파일 사용: {cached_path}")
                return cached_path
        
        try:
            print(f"wandb에서 config 파일 다운로드 중... (run_id: {run_id})")
            
            # wandb API 초기화
            api = wandb.Api()
            
            # run 정보 가져오기
            run = api.run(f"{self.project_name}/{run_id}")
            print(f"Run 이름: {run.name}")
            print(f"Run 상태: {run.state}")
            
            # config 정보 확인
            if hasattr(run, 'config') and run.config:
                print(f"Config 정보:")
                for key, value in run.config.items():
                    print(f"  {key}: {value}")
            
            # config.yaml 파일 찾기
            files = run.files()
            config_file = None
            
            # config.yaml 또는 config.yml 파일 찾기
            for file in files:
                if file.name.endswith(('.yaml', '.yml')) and 'config' in file.name.lower():
                    config_file = file
                    break
            
            if config_file:
                # config 파일 다운로드
                output_path = os.path.join(self.download_dir, f"{run_id}_config.yaml")
                print(f"Config 파일 다운로드 중: {config_file.name} -> {output_path}")
                config_file.download(output_path)
                
                # 메타데이터 업데이트
                file_hash = self._get_file_hash(output_path)
                self.metadata[metadata_key] = {
                    'file_path': output_path,
                    'file_hash': file_hash,
                    'run_id': run_id,
                    'filename': config_file.name,
                    'run_name': run.name,
                    'download_time': str(Path(output_path).stat().st_mtime),
                    'file_size': config_file.size,
                    'type': 'config_file'
                }
                
            else:
                # config 파일이 없으면 run.config를 yaml로 저장
                import yaml
                output_path = os.path.join(self.download_dir, f"{run_id}_config.yaml")
                print(f"Config 파일이 없어서 run.config를 yaml로 저장: {output_path}")
                
                with open(output_path, 'w') as f:
                    yaml.dump(run.config, f, default_flow_style=False, allow_unicode=True)
                
                # 메타데이터 업데이트
                file_hash = self._get_file_hash(output_path)
                self.metadata[metadata_key] = {
                    'file_path': output_path,
                    'file_hash': file_hash,
                    'run_id': run_id,
                    'filename': f"{run_id}_config.yaml",
                    'run_name': run.name,
                    'download_time': str(Path(output_path).stat().st_mtime),
                    'file_size': os.path.getsize(output_path),
                    'type': 'generated_config'
                }
            
            self._save_metadata()
            print(f"Config 파일 다운로드/생성 완료: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"wandb에서 config 다운로드 실패: {e}")
            print(f"\nwandb 인증이 필요할 수 있습니다:")
            print(f"1. wandb login 명령어로 로그인")
            print(f"2. WANDB_API_KEY 환경변수 설정")
            print(f"3. 또는 ~/.netrc 파일에 API 키 설정")
            raise
    
    def get_file_info(self, run_id: str, file_type: str = "model") -> Optional[Dict[str, Any]]:
        """다운로드된 파일 정보를 가져옵니다."""
        if file_type == "model":
            for key, info in self.metadata.items():
                if key.startswith("model_") and info.get('run_id') == run_id:
                    return info
        elif file_type == "config":
            for key, info in self.metadata.items():
                if key.startswith("config_") and info.get('run_id') == run_id:
                    return info
        return None
    
    def list_downloaded_files(self) -> Dict[str, Any]:
        """다운로드된 모든 파일 목록을 반환합니다."""
        return self.metadata
    
    def cleanup_invalid_files(self):
        """유효하지 않은 파일들을 메타데이터에서 제거합니다."""
        invalid_keys = []
        
        for key, info in self.metadata.items():
            file_path = info.get('file_path', '')
            if not self._is_file_valid(file_path, info.get('file_hash')):
                invalid_keys.append(key)
                print(f"유효하지 않은 파일 제거: {file_path}")
        
        for key in invalid_keys:
            del self.metadata[key]
        
        if invalid_keys:
            self._save_metadata()
            print(f"{len(invalid_keys)}개의 유효하지 않은 파일을 메타데이터에서 제거했습니다.")
    
    def force_redownload(self, run_id: str, file_type: str = "model"):
        """특정 파일을 강제로 재다운로드합니다."""
        if file_type == "model":
            # 모델 파일들 찾기
            for key in list(self.metadata.keys()):
                if key.startswith("model_") and self.metadata[key].get('run_id') == run_id:
                    del self.metadata[key]
            print(f"{run_id}의 모델 파일들을 강제 재다운로드하도록 설정했습니다.")
        elif file_type == "config":
            # config 파일 찾기
            for key in list(self.metadata.keys()):
                if key.startswith("config_") and self.metadata[key].get('run_id') == run_id:
                    del self.metadata[key]
            print(f"{run_id}의 config 파일을 강제 재다운로드하도록 설정했습니다.")
        
        self._save_metadata()

# 편의 함수들
def download_model_from_wandb(run_id: str, filename: str = "best_model.pth", 
                             download_dir: str = "downloaded_models") -> str:
    """모델 파일을 다운로드하는 편의 함수"""
    downloader = WandbDownloader(download_dir=download_dir)
    return downloader.download_model(run_id, filename)

def download_config_from_wandb(run_id: str, download_dir: str = "downloaded_configs") -> str:
    """config 파일을 다운로드하는 편의 함수"""
    downloader = WandbDownloader(download_dir=download_dir)
    return downloader.download_config(run_id)

def get_model_path(model_name: str, model_path: str, run_id: str, 
                  download_dir: str = "downloaded_models") -> str:
    """모델 경로를 확인하고, 없으면 wandb에서 다운로드합니다."""
    if model_path and os.path.exists(model_path):
        print(f"{model_name} 모델 파일이 로컬에 존재합니다: {model_path}")
        return model_path
    
    if run_id:
        try:
            downloaded_path = download_model_from_wandb(run_id, output_dir=download_dir)
            print(f"{model_name} 모델을 wandb에서 다운로드했습니다: {downloaded_path}")
            return downloaded_path
        except Exception as e:
            print(f"{model_name} 모델 다운로드 실패: {e}")
            if model_path:
                print(f"로컬 경로도 존재하지 않습니다: {model_path}")
            raise FileNotFoundError(f"{model_name} 모델을 찾을 수 없습니다.")
    
    if model_path:
        raise FileNotFoundError(f"{model_name} 모델 파일을 찾을 수 없습니다: {model_path}")
    else:
        raise ValueError(f"{model_name} 모델의 경로나 run_id가 제공되지 않았습니다.")

def get_config_path(cfg_path: str, run_id: str, download_dir: str = "downloaded_configs") -> str:
    """config 파일 경로를 확인하고, 없으면 wandb에서 다운로드합니다."""
    if cfg_path and os.path.exists(cfg_path):
        print(f"Config 파일이 로컬에 존재합니다: {cfg_path}")
        return cfg_path
    
    if run_id:
        try:
            downloaded_path = download_config_from_wandb(run_id, output_dir=download_dir)
            print(f"Config 파일을 wandb에서 다운로드했습니다: {downloaded_path}")
            return downloaded_path
        except Exception as e:
            print(f"Config 파일 다운로드 실패: {e}")
            if cfg_path:
                print(f"로컬 경로도 존재하지 않습니다: {cfg_path}")
            raise FileNotFoundError(f"Config 파일을 찾을 수 없습니다.")
    
    if cfg_path:
        raise FileNotFoundError(f"Config 파일을 찾을 수 없습니다: {cfg_path}")
    else:
        raise ValueError(f"Config 파일의 경로나 run_id가 제공되지 않았습니다.")
