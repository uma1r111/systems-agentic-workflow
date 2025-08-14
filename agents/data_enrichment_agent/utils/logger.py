import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from enum import Enum

class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"

class EnrichmentLogger:
    def __init__(self, log_dir: str = "agents/data_enrichment_agent/outputs"):
        """
        Initialize Enrichment Logger
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file paths
        self.session_log_file = self.log_dir / "enrichment_session.json"
        self.api_usage_log_file = self.log_dir / "api_usage_log.json"
        self.error_log_file = self.log_dir / "errors.json"
        
        # Initialize session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()
        
        # In-memory session logs
        self.session_logs = []
        self.api_usage_logs = []
        self.error_logs = []
        
        # Setup Python logging for console output
        self._setup_console_logging()
        
        # Initialize session log
        self._initialize_session()
    
    def _setup_console_logging(self):
        """Setup console logging for real-time feedback"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_dir / "console.log")
            ]
        )
        self.console_logger = logging.getLogger("DataEnrichmentAgent")
    
    def _initialize_session(self):
        """Initialize session tracking"""
        self.session_info = {
            "session_id": self.session_id,
            "start_time": self.session_start.isoformat(),
            "end_time": None,
            "status": "running",
            "features_processed": [],
            "total_apis_called": 0,
            "successful_fetches": 0,
            "failed_fetches": 0,
            "total_records_fetched": 0
        }
    
    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log informational message"""
        self._log_message(LogLevel.INFO, message, context)
        self.console_logger.info(message)
    
    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self._log_message(LogLevel.WARNING, message, context)
        self.console_logger.warning(message)
    
    def log_error(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log error message"""
        self._log_message(LogLevel.ERROR, message, context)
        self.console_logger.error(message)
        
        # Also add to error log
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "error_message": message,
            "context": context or {},
            "error_type": "enrichment_error"
        }
        self.error_logs.append(error_entry)
    
    def log_success(self, feature_name: str, api_used: str, api_type: str, 
                   record_count: int, context: Optional[Dict[str, Any]] = None):
        """Log successful data fetch"""
        message = f"Successfully fetched {record_count} records for {feature_name} using {api_type} API: {api_used}"
        
        success_context = {
            "feature_name": feature_name,
            "api_used": api_used,
            "api_type": api_type,
            "record_count": record_count,
            "status": "success"
        }
        
        if context:
            success_context.update(context)
        
        self._log_message(LogLevel.SUCCESS, message, success_context)
        self.console_logger.info(f"✓ {message}")
        
        # Update session statistics
        self.session_info["successful_fetches"] += 1
        self.session_info["total_records_fetched"] += record_count
        
        # Log API usage
        self._log_api_usage(feature_name, api_used, api_type, "success", record_count)
    
    def log_failure(self, feature_name: str, reason: str, context: Optional[Dict[str, Any]] = None):
        """Log failed data fetch"""
        message = f"Failed to fetch data for {feature_name}: {reason}"
        
        failure_context = {
            "feature_name": feature_name,
            "failure_reason": reason,
            "status": "failed"
        }
        
        if context:
            failure_context.update(context)
        
        self._log_message(LogLevel.ERROR, message, failure_context)
        self.console_logger.error(f"✗ {message}")
        
        # Update session statistics
        self.session_info["failed_fetches"] += 1
        
        # Log failed API usage
        self._log_api_usage(feature_name, "unknown", "all_failed", "failed", 0, reason)
    
    def log_api_discovery(self, feature_name: str, apis_found: int, 
                         primary_api: Optional[str] = None, backup_apis: Optional[List[str]] = None):
        """Log API discovery results"""
        message = f"API discovery for {feature_name}: found {apis_found} candidate APIs"
        
        context = {
            "feature_name": feature_name,
            "apis_found": apis_found,
            "primary_api": primary_api,
            "backup_apis": backup_apis or [],
            "discovery_status": "completed"
        }
        
        self._log_message(LogLevel.INFO, message, context)
        self.console_logger.info(message)
    
    def log_validation_result(self, feature_name: str, validation_summary: Dict[str, Any]):
        """Log data validation results"""
        status = validation_summary.get("status", "unknown")
        total_records = validation_summary.get("total_records", 0)
        valid_records = validation_summary.get("valid_records", 0)
        completeness = validation_summary.get("completeness", "0%")
        
        message = f"Data validation for {feature_name}: {status} - {valid_records}/{total_records} valid records ({completeness})"
        
        context = {
            "feature_name": feature_name,
            "validation_summary": validation_summary,
            "validation_status": status
        }
        
        level = LogLevel.SUCCESS if status == "success" else LogLevel.WARNING
        self._log_message(level, message, context)
        
        if status == "success":
            self.console_logger.info(f"✓ {message}")
        else:
            self.console_logger.warning(f"⚠ {message}")
    
    def _log_message(self, level: LogLevel, message: str, context: Optional[Dict[str, Any]] = None):
        """Internal method to log structured message"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "level": level.value,
            "message": message,
            "context": context or {}
        }
        
        self.session_logs.append(log_entry)
    
    def _log_api_usage(self, feature_name: str, api_name: str, api_type: str, 
                      status: str, record_count: int, error_reason: Optional[str] = None):
        """Log API usage statistics"""
        usage_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "feature_name": feature_name,
            "api_name": api_name,
            "api_type": api_type,  # primary, backup_1, backup_2
            "status": status,  # success, failed, timeout, rate_limited
            "record_count": record_count,
            "error_reason": error_reason,
            "response_time_seconds": None  # Could be enhanced to track this
        }
        
        self.api_usage_logs.append(usage_entry)
        self.session_info["total_apis_called"] += 1
    
    def start_feature_processing(self, feature_name: str):
        """Mark start of feature processing"""
        self.log_info(f"Starting processing for feature: {feature_name}")
        
        if feature_name not in self.session_info["features_processed"]:
            self.session_info["features_processed"].append(feature_name)
    
    def end_session(self, final_status: str = "completed"):
        """End logging session and write all logs to files"""
        self.session_info["end_time"] = datetime.now().isoformat()
        self.session_info["status"] = final_status
        self.session_info["duration_minutes"] = round(
            (datetime.now() - self.session_start).total_seconds() / 60, 2
        )
        
        # Calculate success rate
        total_attempts = self.session_info["successful_fetches"] + self.session_info["failed_fetches"]
        if total_attempts > 0:
            self.session_info["success_rate"] = round(
                (self.session_info["successful_fetches"] / total_attempts) * 100, 1
            )
        else:
            self.session_info["success_rate"] = 0
        
        # Write all logs to files
        self._write_logs_to_files()
        
        # Final console summary
        self._print_session_summary()
    
    def _write_logs_to_files(self):
        """Write all accumulated logs to JSON files"""
        try:
            # Write session log
            session_data = {
                "session_info": self.session_info,
                "logs": self.session_logs
            }
            
            with open(self.session_log_file, "w") as f:
                json.dump(session_data, f, indent=2, default=str)
            
            # Write API usage log
            with open(self.api_usage_log_file, "w") as f:
                json.dump(self.api_usage_logs, f, indent=2, default=str)
            
            # Write error log (only if there are errors)
            if self.error_logs:
                with open(self.error_log_file, "w") as f:
                    json.dump(self.error_logs, f, indent=2, default=str)
            
            self.console_logger.info(f"Logs written to {self.log_dir}")
            
        except Exception as e:
            self.console_logger.error(f"Failed to write logs to files: {str(e)}")
    
    def _print_session_summary(self):
        """Print session summary to console"""
        print("\n" + "="*60)
        print("DATA ENRICHMENT SESSION SUMMARY")
        print("="*60)
        print(f"Session ID: {self.session_id}")
        print(f"Duration: {self.session_info['duration_minutes']} minutes")
        print(f"Status: {self.session_info['status']}")
        print()
        print("PROCESSING RESULTS:")
        print(f"  Features Processed: {len(self.session_info['features_processed'])}")
        print(f"  Successful Fetches: {self.session_info['successful_fetches']}")
        print(f"  Failed Fetches: {self.session_info['failed_fetches']}")
        print(f"  Success Rate: {self.session_info['success_rate']}%")
        print(f"  Total Records Fetched: {self.session_info['total_records_fetched']:,}")
        print()
        print("API USAGE:")
        print(f"  Total API Calls: {self.session_info['total_apis_called']}")
        
        # API breakdown
        api_stats = {}
        for usage in self.api_usage_logs:
            api_name = usage['api_name']
            status = usage['status']
            if api_name not in api_stats:
                api_stats[api_name] = {'success': 0, 'failed': 0}
            api_stats[api_name][status] += 1
        
        if api_stats:
            print("  API Breakdown:")
            for api_name, stats in api_stats.items():
                total = stats['success'] + stats['failed']
                success_rate = (stats['success'] / total * 100) if total > 0 else 0
                print(f"    {api_name}: {stats['success']}/{total} success ({success_rate:.1f}%)")
        
        print("\nFeatures Processed:")
        for feature in self.session_info["features_processed"]:
            print(f"  - {feature}")
        
        print(f"\nLog files saved to: {self.log_dir}")
        print("="*60)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return self.session_info.copy()
    
    def get_api_usage_stats(self) -> List[Dict[str, Any]]:
        """Get API usage statistics"""
        return self.api_usage_logs.copy()
    
    def export_logs(self, export_path: str):
        """Export all logs to a single JSON file"""
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "session_info": self.session_info,
                "session_logs": self.session_logs,
                "api_usage_logs": self.api_usage_logs,
                "error_logs": self.error_logs
            }
            
            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.log_info(f"Logs exported to {export_path}")
            
        except Exception as e:
            self.log_error(f"Failed to export logs: {str(e)}")