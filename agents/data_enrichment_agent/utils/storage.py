import pandas as pd
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import shutil

class DataStorage:
    def __init__(self, base_data_dir: str = "data"):
        """
        Initialize Data Storage handler
        
        Args:
            base_data_dir: Base directory for all data storage
        """
        self.base_data_dir = Path(base_data_dir)
        
        # Create directory structure
        self.raw_features_dir = self.base_data_dir / "raw_features"
        self.enriched_dir = self.base_data_dir / "enriched" 
        self.metadata_dir = self.base_data_dir / "metadata"
        self.archive_dir = self.base_data_dir / "archive"
        
        # Create all directories
        for directory in [self.raw_features_dir, self.enriched_dir, 
                         self.metadata_dir, self.archive_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Storage configuration
        self.max_file_size_mb = 100  # Maximum file size before compression
        self.retention_days = 90     # Days to keep archived data
        
    def save_feature_data(self, df: pd.DataFrame, feature_name: str, 
                         city: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save feature data as CSV with metadata
        
        Args:
            df: DataFrame containing feature data
            feature_name: Name of the feature
            city: City name for data organization
            metadata: Optional metadata dictionary
            
        Returns:
            Path to saved file
        """
        if df is None or df.empty:
            raise ValueError("Cannot save empty DataFrame")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{feature_name}_{city.lower()}_{timestamp}.csv"
        filepath = self.raw_features_dir / filename
        
        try:
            # Save CSV data
            df.to_csv(filepath, index=True)
            
            # Save metadata
            self._save_feature_metadata(filepath, feature_name, city, df, metadata)
            
            # Check file size and compress if needed
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                compressed_path = self._compress_file(filepath)
                if compressed_path:
                    filepath = compressed_path
            
            return str(filepath)
            
        except Exception as e:
            # Clean up partial files on error
            if filepath.exists():
                filepath.unlink()
            raise Exception(f"Failed to save feature data: {str(e)}")
    
    def _save_feature_metadata(self, data_filepath: Path, feature_name: str, 
                              city: str, df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None):
        """Save metadata for feature data file"""
        metadata_filename = data_filepath.stem + "_metadata.json"
        metadata_filepath = self.metadata_dir / metadata_filename
        
        # Generate file hash for integrity checking
        file_hash = self._calculate_file_hash(data_filepath)
        
        # Compile metadata
        file_metadata = {
            "feature_name": feature_name,
            "city": city,
            "data_file": str(data_filepath),
            "created_timestamp": datetime.now().isoformat(),
            "file_hash": file_hash,
            "file_size_bytes": data_filepath.stat().st_size,
            "data_info": {
                "total_records": len(df),
                "columns": list(df.columns),
                "date_range": {
                    "start": str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) else None,
                    "end": str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None
                },
                "data_types": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "basic_stats": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {}
            }
        }
        
        # Add custom metadata if provided
        if metadata:
            file_metadata["custom_metadata"] = metadata
        
        # Save metadata
        with open(metadata_filepath, "w") as f:
            json.dump(file_metadata, f, indent=2, default=str)
    
    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate MD5 hash of file for integrity checking"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _compress_file(self, filepath: Path) -> Optional[Path]:
        """Compress large files using gzip"""
        try:
            import gzip
            
            compressed_path = filepath.with_suffix(filepath.suffix + '.gz')
            
            with open(filepath, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file after successful compression
            filepath.unlink()
            
            return compressed_path
            
        except Exception as e:
            print(f"Failed to compress file {filepath}: {str(e)}")
            return None
    
    def consolidate_enriched_data(self, session_results: List[Dict[str, Any]], 
                                output_filename: str = None) -> str:
        """
        Consolidate all successfully fetched features into single enriched dataset
        
        Args:
            session_results: List of processing results from enrichment session
            output_filename: Optional custom filename for consolidated data
            
        Returns:
            Path to consolidated enriched dataset
        """
        successful_results = [r for r in session_results if r.get("status") == "success"]
        
        if not successful_results:
            raise ValueError("No successful feature data to consolidate")
        
        consolidated_data = []
        feature_info = []
        
        # Load and merge all feature data
        for result in successful_results:
            data_file = result.get("data_file")
            feature_name = result.get("feature_name")
            
            if not data_file or not os.path.exists(data_file):
                continue
            
            try:
                # Load feature data
                if data_file.endswith('.gz'):
                    df = pd.read_csv(data_file, compression='gzip', index_col=0, parse_dates=True)
                else:
                    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                
                # Add to consolidation list
                consolidated_data.append(df)
                
                feature_info.append({
                    "feature_name": feature_name,
                    "records": len(df),
                    "date_range": {
                        "start": str(df.index.min()),
                        "end": str(df.index.max())
                    },
                    "source_file": data_file
                })
                
            except Exception as e:
                print(f"Failed to load feature data from {data_file}: {str(e)}")
                continue
        
        if not consolidated_data:
            raise ValueError("No feature data could be loaded for consolidation")
        
        # Merge all features on datetime index
        enriched_df = self._merge_feature_dataframes(consolidated_data)
        
        # Generate output filename
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"enriched_data_{timestamp}.csv"
        
        output_path = self.enriched_dir / output_filename
        
        # Save consolidated data
        enriched_df.to_csv(output_path)
        
        # Save consolidation metadata
        self._save_consolidation_metadata(output_path, feature_info, enriched_df)
        
        return str(output_path)
    
    def _merge_feature_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple feature DataFrames on datetime index"""
        if len(dataframes) == 1:
            return dataframes[0]
        
        # Start with first DataFrame
        merged_df = dataframes[0]
        
        # Merge each additional DataFrame
        for df in dataframes[1:]:
            merged_df = pd.merge(
                merged_df, df,
                left_index=True, right_index=True,
                how='outer',
                suffixes=('', '_dup')
            )
            
            # Remove duplicate columns (keep original)
            duplicate_cols = [col for col in merged_df.columns if col.endswith('_dup')]
            merged_df.drop(columns=duplicate_cols, inplace=True)
        
        # Sort by datetime index
        merged_df.sort_index(inplace=True)
        
        return merged_df
    
    def _save_consolidation_metadata(self, output_path: Path, feature_info: List[Dict[str, Any]], 
                                   merged_df: pd.DataFrame):
        """Save metadata for consolidated enriched dataset"""
        metadata_filename = output_path.stem + "_consolidation_metadata.json"
        metadata_path = self.metadata_dir / metadata_filename
        
        consolidation_metadata = {
            "consolidated_file": str(output_path),
            "created_timestamp": datetime.now().isoformat(),
            "total_features": len(feature_info),
            "total_records": len(merged_df),
            "date_range": {
                "start": str(merged_df.index.min()),
                "end": str(merged_df.index.max())
            },
            "features_included": feature_info,
            "data_completeness": {
                "total_possible_values": merged_df.size,
                "non_null_values": merged_df.count().sum(),
                "completeness_percentage": round((merged_df.count().sum() / merged_df.size) * 100, 2)
            },
            "feature_completeness": {
                col: {
                    "non_null_count": merged_df[col].count(),
                    "completeness_percentage": round((merged_df[col].count() / len(merged_df)) * 100, 2)
                }
                for col in merged_df.columns
            }
        }
        
        with open(metadata_path, "w") as f:
            json.dump(consolidation_metadata, f, indent=2, default=str)
    
    def get_available_features(self, city: Optional[str] = None, 
                              days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get list of available feature data files
        
        Args:
            city: Filter by city (optional)
            days_back: Only show files from last N days (optional)
            
        Returns:
            List of available feature files with metadata
        """
        available_features = []
        
        # Get all CSV files in raw_features directory
        for filepath in self.raw_features_dir.glob("*.csv*"):
            # Get corresponding metadata
            metadata_filename = filepath.stem.replace('.gz', '') + "_metadata.json"
            metadata_path = self.metadata_dir / metadata_filename
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    # Apply city filter
                    if city and metadata.get("city", "").lower() != city.lower():
                        continue
                    
                    # Apply date filter
                    if days_back:
                        created_date = datetime.fromisoformat(metadata.get("created_timestamp", ""))
                        cutoff_date = datetime.now() - pd.Timedelta(days=days_back)
                        if created_date < cutoff_date:
                            continue
                    
                    available_features.append({
                        "filepath": str(filepath),
                        "feature_name": metadata.get("feature_name"),
                        "city": metadata.get("city"),
                        "created_timestamp": metadata.get("created_timestamp"),
                        "records": metadata.get("data_info", {}).get("total_records", 0),
                        "file_size_mb": round(metadata.get("file_size_bytes", 0) / (1024*1024), 2)
                    })
                    
                except Exception as e:
                    print(f"Failed to read metadata for {filepath}: {str(e)}")
                    continue
        
        # Sort by creation time (newest first)
        available_features.sort(key=lambda x: x["created_timestamp"], reverse=True)
        
        return available_features
    
    def cleanup_old_files(self, retention_days: Optional[int] = None):
        """
        Clean up old data files beyond retention period
        
        Args:
            retention_days: Days to retain files (uses instance default if None)
        """
        retention_days = retention_days or self.retention_days
        cutoff_date = datetime.now() - pd.Timedelta(days=retention_days)
        
        archived_files = []
        
        # Check all files in raw_features directory
        for filepath in self.raw_features_dir.glob("*"):
            file_date = datetime.fromtimestamp(filepath.stat().st_mtime)
            
            if file_date < cutoff_date:
                # Move to archive directory
                archive_path = self.archive_dir / filepath.name
                shutil.move(str(filepath), str(archive_path))
                archived_files.append(str(filepath))
                
                # Also move corresponding metadata
                metadata_filename = filepath.stem.replace('.gz', '') + "_metadata.json"
                metadata_path = self.metadata_dir / metadata_filename
                if metadata_path.exists():
                    archive_metadata_path = self.archive_dir / metadata_path.name
                    shutil.move(str(metadata_path), str(archive_metadata_path))
        
        return archived_files
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """Get summary of storage usage and file counts"""
        summary = {
            "directories": {
                "raw_features": str(self.raw_features_dir),
                "enriched": str(self.enriched_dir),
                "metadata": str(self.metadata_dir),
                "archive": str(self.archive_dir)
            },
            "file_counts": {},
            "storage_usage_mb": {},
            "total_storage_mb": 0
        }
        
        # Calculate storage for each directory
        for name, directory in summary["directories"].items():
            dir_path = Path(directory)
            if dir_path.exists():
                files = list(dir_path.glob("*"))
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                
                summary["file_counts"][name] = len(files)
                summary["storage_usage_mb"][name] = round(total_size / (1024*1024), 2)
                summary["total_storage_mb"] += summary["storage_usage_mb"][name]
        
        summary["total_storage_mb"] = round(summary["total_storage_mb"], 2)
        
        return summary
    
    def verify_data_integrity(self, filepath: str) -> bool:
        """
        Verify data file integrity using stored hash
        
        Args:
            filepath: Path to data file to verify
            
        Returns:
            True if file integrity is verified, False otherwise
        """
        data_path = Path(filepath)
        if not data_path.exists():
            return False
        
        # Find corresponding metadata
        metadata_filename = data_path.stem.replace('.gz', '') + "_metadata.json"
        metadata_path = self.metadata_dir / metadata_filename
        
        if not metadata_path.exists():
            print(f"No metadata found for {filepath}")
            return False
        
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            stored_hash = metadata.get("file_hash")
            if not stored_hash:
                print(f"No hash found in metadata for {filepath}")
                return False
            
            # Calculate current hash
            current_hash = self._calculate_file_hash(data_path)
            
            # Compare hashes
            if current_hash == stored_hash:
                return True
            else:
                print(f"Hash mismatch for {filepath}: expected {stored_hash}, got {current_hash}")
                return False
                
        except Exception as e:
            print(f"Error verifying integrity of {filepath}: {str(e)}")
            return False
    
    def backup_critical_data(self, backup_dir: str) -> List[str]:
        """
        Create backup of critical enriched datasets
        
        Args:
            backup_dir: Directory to store backups
            
        Returns:
            List of backed up files
        """
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        backed_up_files = []
        
        # Backup enriched datasets (most recent ones)
        enriched_files = sorted(
            self.enriched_dir.glob("enriched_data_*.csv*"), 
            key=lambda x: x.stat().st_mtime, 
            reverse=True
        )
        
        # Backup last 5 enriched datasets
        for filepath in enriched_files[:5]:
            backup_file = backup_path / filepath.name
            shutil.copy2(str(filepath), str(backup_file))
            backed_up_files.append(str(backup_file))
            
            # Also backup corresponding metadata
            metadata_filename = filepath.stem.replace('.gz', '') + "_consolidation_metadata.json"
            metadata_path = self.metadata_dir / metadata_filename
            if metadata_path.exists():
                backup_metadata = backup_path / metadata_path.name
                shutil.copy2(str(metadata_path), str(backup_metadata))
                backed_up_files.append(str(backup_metadata))
        
        return backed_up_files
    
    def load_feature_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Load feature data from file with error handling
        
        Args:
            filepath: Path to feature data file
            
        Returns:
            DataFrame or None if loading fails
        """
        try:
            if filepath.endswith('.gz'):
                df = pd.read_csv(filepath, compression='gzip', index_col=0, parse_dates=True)
            else:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            return df
            
        except Exception as e:
            print(f"Failed to load data from {filepath}: {str(e)}")
            return None
    
    def get_feature_metadata(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a feature data file
        
        Args:
            filepath: Path to feature data file
            
        Returns:
            Metadata dictionary or None if not found
        """
        data_path = Path(filepath)
        metadata_filename = data_path.stem.replace('.gz', '') + "_metadata.json"
        metadata_path = self.metadata_dir / metadata_filename
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load metadata from {metadata_path}: {str(e)}")
            return None
    
    def create_data_catalog(self) -> str:
        """
        Create a comprehensive catalog of all stored data
        
        Returns:
            Path to created catalog file
        """
        catalog_data = {
            "catalog_created": datetime.now().isoformat(),
            "storage_summary": self.get_storage_summary(),
            "raw_features": [],
            "enriched_datasets": [],
            "archived_data": []
        }
        
        # Catalog raw features
        for filepath in self.raw_features_dir.glob("*.csv*"):
            metadata = self.get_feature_metadata(str(filepath))
            catalog_entry = {
                "filepath": str(filepath),
                "filename": filepath.name,
                "file_size_mb": round(filepath.stat().st_size / (1024*1024), 2),
                "modified_date": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
            }
            
            if metadata:
                catalog_entry.update({
                    "feature_name": metadata.get("feature_name"),
                    "city": metadata.get("city"),
                    "records": metadata.get("data_info", {}).get("total_records", 0),
                    "date_range": metadata.get("data_info", {}).get("date_range", {})
                })
            
            catalog_data["raw_features"].append(catalog_entry)
        
        # Catalog enriched datasets
        for filepath in self.enriched_dir.glob("*.csv*"):
            catalog_entry = {
                "filepath": str(filepath),
                "filename": filepath.name,
                "file_size_mb": round(filepath.stat().st_size / (1024*1024), 2),
                "modified_date": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
            }
            
            # Try to get consolidation metadata
            metadata_filename = filepath.stem.replace('.gz', '') + "_consolidation_metadata.json"
            metadata_path = self.metadata_dir / metadata_filename
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        consolidation_metadata = json.load(f)
                    
                    catalog_entry.update({
                        "total_features": consolidation_metadata.get("total_features"),
                        "total_records": consolidation_metadata.get("total_records"),
                        "date_range": consolidation_metadata.get("date_range", {}),
                        "completeness_percentage": consolidation_metadata.get("data_completeness", {}).get("completeness_percentage")
                    })
                except Exception:
                    pass
            
            catalog_data["enriched_datasets"].append(catalog_entry)
        
        # Catalog archived data
        for filepath in self.archive_dir.glob("*.csv*"):
            catalog_entry = {
                "filepath": str(filepath),
                "filename": filepath.name,
                "file_size_mb": round(filepath.stat().st_size / (1024*1024), 2),
                "archived_date": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
            }
            catalog_data["archived_data"].append(catalog_entry)
        
        # Sort entries by date (newest first)
        for category in ["raw_features", "enriched_datasets", "archived_data"]:
            catalog_data[category].sort(
                key=lambda x: x.get("modified_date", x.get("archived_date", "")), 
                reverse=True
            )
        
        # Save catalog
        catalog_path = self.metadata_dir / f"data_catalog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(catalog_path, "w") as f:
            json.dump(catalog_data, f, indent=2, default=str)
        
        return str(catalog_path)
    
    def export_data_for_analysis(self, features: List[str], 
                                date_range: Optional[Tuple[str, str]] = None,
                                output_format: str = "csv") -> str:
        """
        Export specific features for external analysis
        
        Args:
            features: List of feature names to export
            date_range: Optional tuple of (start_date, end_date) in YYYY-MM-DD format
            output_format: Export format ('csv', 'json', 'parquet')
            
        Returns:
            Path to exported file
        """
        # Find and load feature data
        feature_dataframes = []
        
        for feature_name in features:
            # Find most recent file for this feature
            feature_files = sorted(
                [f for f in self.raw_features_dir.glob(f"{feature_name}_*.csv*")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if feature_files:
                df = self.load_feature_data(str(feature_files[0]))
                if df is not None:
                    feature_dataframes.append(df)
        
        if not feature_dataframes:
            raise ValueError(f"No data found for features: {features}")
        
        # Merge features
        export_df = self._merge_feature_dataframes(feature_dataframes)
        
        # Apply date range filter if specified
        if date_range and isinstance(export_df.index, pd.DatetimeIndex):
            start_date, end_date = date_range
            export_df = export_df.loc[start_date:end_date]
        
        # Generate export filename
        features_str = "_".join(features[:3])  # Use first 3 feature names
        if len(features) > 3:
            features_str += f"_and_{len(features)-3}_more"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"export_{features_str}_{timestamp}.{output_format}"
        export_path = self.enriched_dir / filename
        
        # Export in specified format
        if output_format.lower() == "csv":
            export_df.to_csv(export_path)
        elif output_format.lower() == "json":
            export_df.to_json(export_path, orient="index", date_format="iso")
        elif output_format.lower() == "parquet":
            export_df.to_parquet(export_path)
        else:
            raise ValueError(f"Unsupported export format: {output_format}")
        
        return str(export_path)