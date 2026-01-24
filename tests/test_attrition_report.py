"""Tests for the data attrition report in stage_get_data."""

import argparse
import importlib.util
import logging
from io import StringIO


def _load_attrition_report_function():
    """Load the _log_data_attrition_report function from stage_get_data module."""
    spec = importlib.util.spec_from_file_location(
        "stage_get_data", 
        "utils/01_get_data/stage_get_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._log_data_attrition_report


def test_log_data_attrition_report_formats_correctly():
    """Test that the attrition report logs without errors and contains key sections."""
    _log_data_attrition_report = _load_attrition_report_function()
    
    # Create mock stats similar to real output
    mock_stats = {
        'likes': {
            'n_likes_initial': 154984346,
            'n_users_initial': 2478580,
            'n_users_eligible_for_sampling': 2064343,
            'n_users_excluded_min_likes': 414237,
            'n_users_sampled': 200000,
            'n_likes_after_user_sample': 15030063,
            'n_likes_after_per_user_cap': 3879155,
            'n_likes_final': 3844144,
            'n_users_final': 135920,
            'n_likes_removed_by_join': 1196976,
            'n_users_removed_by_join_verify': 13752,
            'n_users_final_after_join': 115680,
            'n_likes_final_after_join': 2633416,
        },
        'posts': {
            'n_posts_total': 41415426,
            'n_liked_posts': 1349077,
            'n_liked_only': 1344901,
            'n_liked_in_random_sample': 4176,
            'liked_post_match_rate': 69.58,
            'n_random_sample': 100000,
            'n_posts_core': 1444901,
        },
        'memory_actual': {
            'n_checkpoints': 6,
            'peak_process_gb': 62.48,
            'start_process_gb': 0.73,
            'end_process_gb': 62.48,
            'checkpoints': [
                {'name': 'pipeline_start', 'elapsed_sec': 0.0, 'process_gb': 0.73},
                {'name': 'after_memory_check', 'elapsed_sec': 243.2, 'process_gb': 0.79},
                {'name': 'after_likes_load', 'elapsed_sec': 500.0, 'process_gb': 32.18},
                {'name': 'after_uri_extraction', 'elapsed_sec': 502.1, 'process_gb': 32.51},
                {'name': 'after_posts_load_and_expansion', 'elapsed_sec': 1356.6, 'process_gb': 62.31},
                {'name': 'after_join_verification', 'elapsed_sec': 1360.5, 'process_gb': 62.48},
            ]
        },
    }
    
    mock_memory_estimate = {
        'estimated_peak_gb': 147.09,
    }
    
    # Create mock args
    mock_args = argparse.Namespace(
        min_likes_per_user=2,
        max_likes_per_user=500,
        max_liking_users=200000,
        negative_posts_sample=100000,
    )
    
    # Capture log output
    log_output = StringIO()
    handler = logging.StreamHandler(log_output)
    handler.setLevel(logging.INFO)
    
    logger = logging.getLogger('test_attrition_report')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # Run the report function
    _log_data_attrition_report(mock_stats, mock_memory_estimate, mock_args, logger)
    
    # Get the logged content
    log_content = log_output.getvalue()
    
    # Verify key sections are present
    assert 'DATA ATTRITION REPORT' in log_content
    assert 'LIKES PIPELINE' in log_content
    assert 'POSTS PIPELINE' in log_content
    assert 'POST-JOIN VERIFICATION' in log_content
    assert 'FINAL OUTPUT' in log_content
    assert 'OVERALL ATTRITION SUMMARY' in log_content
    assert 'MEMORY SUMMARY' in log_content
    
    # Verify some key numbers are present
    assert '154,984,346' in log_content  # Initial likes
    assert '2,478,580' in log_content  # Initial users
    assert '115,680' in log_content  # Final users
    assert '2,633,416' in log_content  # Final likes
    assert '62.48' in log_content  # Peak memory
    
    # Cleanup
    logger.removeHandler(handler)


def test_log_data_attrition_report_handles_empty_stats():
    """Test that the report handles missing/empty stats gracefully."""
    _log_data_attrition_report = _load_attrition_report_function()
    
    # Empty stats
    mock_stats = {}
    mock_memory_estimate = {}
    mock_args = argparse.Namespace(
        min_likes_per_user=2,
        max_likes_per_user=100,
        max_liking_users=0,
        negative_posts_sample=10000,
    )
    
    log_output = StringIO()
    handler = logging.StreamHandler(log_output)
    handler.setLevel(logging.INFO)
    
    logger = logging.getLogger('test_empty_stats')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # Should not raise an exception
    _log_data_attrition_report(mock_stats, mock_memory_estimate, mock_args, logger)
    
    log_content = log_output.getvalue()
    assert 'DATA ATTRITION REPORT' in log_content
    
    logger.removeHandler(handler)


def test_log_data_attrition_report_handles_partial_stats():
    """Test that the report handles partial stats (some sections missing)."""
    _log_data_attrition_report = _load_attrition_report_function()
    
    # Partial stats - only likes section
    mock_stats = {
        'likes': {
            'n_likes_initial': 1000,
            'n_users_initial': 100,
            'n_likes_final': 500,
            'n_users_final': 50,
        },
    }
    mock_memory_estimate = None
    mock_args = argparse.Namespace(
        min_likes_per_user=2,
        max_likes_per_user=100,
        max_liking_users=0,
        negative_posts_sample=10000,
    )
    
    log_output = StringIO()
    handler = logging.StreamHandler(log_output)
    handler.setLevel(logging.INFO)
    
    logger = logging.getLogger('test_partial_stats')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # Should not raise an exception
    _log_data_attrition_report(mock_stats, mock_memory_estimate, mock_args, logger)
    
    log_content = log_output.getvalue()
    assert 'DATA ATTRITION REPORT' in log_content
    assert '1,000' in log_content  # Initial likes
    
    logger.removeHandler(handler)
