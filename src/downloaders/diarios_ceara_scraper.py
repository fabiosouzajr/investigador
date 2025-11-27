#!/usr/bin/env python3
"""
Diários Oficiais Ceará Web Scraper

Downloads PDF files from the Diário Oficial do Estado do Ceará website.
Supports downloading by year or last X days, with progress feedback and retry logic.
"""

import os
import sys
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional
from bs4 import BeautifulSoup
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from urllib.parse import urlparse, parse_qs, unquote
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diarios_ceara_scraper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
BASE_URL_SHORTCUT = "http://pesquisa.doe.seplag.ce.gov.br/doepesquisa/sead.do?page=ultimasDetalhe&cmd=10&action=Cadernos&data="
BASE_URL_MAIN = "http://pesquisa.doe.seplag.ce.gov.br/doepesquisa/sead.do?page=ultimasEdicoes&cmd=11&action=Ultimas"
BASE_DIR = "diarios_ceara"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"


def extract_filename_from_url(url: str) -> Optional[str]:
    """
    Extract filename from URL by checking query parameters, path, and Content-Disposition.
    
    Args:
        url: URL to extract filename from
        
    Returns:
        Filename if found, None otherwise
    """
    try:
        parsed = urlparse(url)
        
        # Check query parameters for common filename parameters
        query_params = parse_qs(parsed.query)
        for param in ['arquivo', 'file', 'filename', 'nome', 'documento']:
            if param in query_params:
                filename = query_params[param][0]
                if filename and filename.lower().endswith('.pdf'):
                    return unquote(filename)
        
        # Check path for PDF filename
        path = parsed.path
        if path:
            # Look for .pdf in the path
            pdf_match = re.search(r'([^/]+\.pdf)', path, re.IGNORECASE)
            if pdf_match:
                return unquote(pdf_match.group(1))
            
            # Get last part of path
            basename = os.path.basename(path)
            if basename and basename.lower().endswith('.pdf'):
                return unquote(basename)
        
        return None
    except Exception as e:
        logger.debug(f"Error extracting filename from URL {url}: {e}")
        return None


def get_filename_from_content_disposition(url: str, headers: dict) -> Optional[str]:
    """
    Try to get filename from Content-Disposition header by making a HEAD request.
    
    Args:
        url: URL to check
        headers: Headers to use for request
        
    Returns:
        Filename if found in Content-Disposition header, None otherwise
    """
    try:
        response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        content_disposition = response.headers.get('Content-Disposition', '')
        if content_disposition:
            # Parse Content-Disposition: attachment; filename="file.pdf"
            filename_match = re.search(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', content_disposition, re.IGNORECASE)
            if filename_match:
                filename = filename_match.group(1).strip('"\'')
                return unquote(filename)
    except Exception:
        pass
    return None


def test_shortcut_url(date_str: str) -> Tuple[bool, List[Tuple[str, str]]]:
    """
    Test if shortcut URL works for a given date.
    
    Args:
        date_str: Date in YYYYMMDD format
        
    Returns:
        Tuple of (success: bool, links: List[Tuple[filename, url]])
    """
    url = BASE_URL_SHORTCUT + date_str
    headers = {'User-Agent': USER_AGENT}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for table with PDF links
        # Common patterns: links in tables, links with .pdf extension, etc.
        links = []
        
        # Try to find links in tables
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                for cell in cells:
                    link = cell.find('a', href=True)
                    if not link:
                        continue
                    
                    href = link.get('href', '')
                    if not ('.pdf' in href.lower() or 'download' in href.lower() or 'baixar' in href.lower()):
                        continue
                    
                    # Make absolute URL if relative
                    if href.startswith('/'):
                        full_url = f"http://pesquisa.doe.seplag.ce.gov.br{href}"
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        full_url = f"http://pesquisa.doe.seplag.ce.gov.br/doepesquisa/{href}"
                    
                    # Try multiple methods to get filename
                    filename = None
                    
                    # Method 1: Extract from URL
                    filename = extract_filename_from_url(full_url)
                    
                    # Method 2: Look in table cells (often filename is in adjacent cell)
                    if not filename or filename.lower() in ['visualizar.pdf', 'visualizar', 'baixar.pdf', 'download.pdf']:
                        for other_cell in cells:
                            cell_text = other_cell.get_text(strip=True)
                            # Skip if it's the link text or common action words
                            if (cell_text and 
                                cell_text.lower() not in ['visualizar', 'baixar', 'download', 'ver'] and
                                (cell_text.lower().endswith('.pdf') or len(cell_text) > 5)):
                                filename = cell_text
                                if not filename.endswith('.pdf'):
                                    filename += '.pdf'
                                break
                    
                    # Method 3: Try to get from Content-Disposition header
                    if not filename or filename.lower() in ['visualizar.pdf', 'visualizar', 'baixar.pdf', 'download.pdf']:
                        filename = get_filename_from_content_disposition(full_url, headers)
                    
                    # Method 4: Use link text if it's not a generic action word
                    if not filename or filename.lower() in ['visualizar.pdf', 'visualizar', 'baixar.pdf', 'download.pdf']:
                        link_text = link.get_text(strip=True)
                        if link_text and link_text.lower() not in ['visualizar', 'baixar', 'download', 'ver']:
                            filename = link_text
                        else:
                            # Fallback: use basename from URL
                            filename = os.path.basename(urlparse(full_url).path) or 'documento.pdf'
                    
                    # Ensure .pdf extension
                    if not filename.endswith('.pdf'):
                        filename += '.pdf'
                    
                    # Clean filename
                    filename = unquote(filename)
                    
                    links.append((filename, full_url))
        
        # Also check for direct PDF links in the page
        if not links:
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                if href.lower().endswith('.pdf'):
                    if href.startswith('/'):
                        full_url = f"http://pesquisa.doe.seplag.ce.gov.br{href}"
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        full_url = f"http://pesquisa.doe.seplag.ce.gov.br/doepesquisa/{href}"
                    
                    # Try to extract filename from URL
                    filename = extract_filename_from_url(full_url)
                    if not filename:
                        filename = os.path.basename(urlparse(full_url).path) or 'documento.pdf'
                    filename = unquote(filename)
                    if not filename.endswith('.pdf'):
                        filename += '.pdf'
                    links.append((filename, full_url))
        
        # Check if page indicates no publications (e.g., "não há publicações", "sem dados")
        page_text = soup.get_text().lower()
        if any(phrase in page_text for phrase in ['não há', 'sem dados', 'nenhuma publicação', 'não existem']):
            return True, []  # Page loaded but no publications
        
        return True, links
        
    except requests.RequestException as e:
        logger.error(f"Error testing shortcut URL for {date_str}: {e}")
        return False, []
    except Exception as e:
        logger.error(f"Unexpected error testing shortcut URL for {date_str}: {e}")
        return False, []


def get_user_input() -> Tuple[str, int]:
    """
    Get user input for download mode and value.
    
    Returns:
        Tuple of (mode: 'year' or 'days', value: int)
    """
    while True:
        try:
            choice = input("Download by (1) Year or (2) Last X days? Enter 1 or 2: ").strip()
            if choice == '1':
                year = input("Enter year (e.g., 2024): ").strip()
                year_int = int(year)
                if 2000 <= year_int <= datetime.now().year + 1:
                    return ('year', year_int)
                else:
                    print(f"Please enter a valid year between 2000 and {datetime.now().year + 1}")
            elif choice == '2':
                days = input("Enter number of days: ").strip()
                days_int = int(days)
                if days_int > 0:
                    return ('days', days_int)
                else:
                    print("Please enter a positive number of days")
            else:
                print("Please enter 1 or 2")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(0)


def generate_dates(mode: str, value: int) -> List[str]:
    """
    Generate list of dates in YYYYMMDD format.
    
    Args:
        mode: 'year' or 'days'
        value: Year number or number of days
        
    Returns:
        List of date strings in YYYYMMDD format
    """
    dates = []
    
    if mode == 'year':
        # Generate all dates in the year
        start_date = datetime(value, 1, 1)
        # Check if year is current year
        if value == datetime.now().year:
            end_date = datetime.now()
        else:
            end_date = datetime(value, 12, 31)
        
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date.strftime('%Y%m%d'))
            current_date += timedelta(days=1)
    
    elif mode == 'days':
        # Generate last X days from today
        today = datetime.now()
        for i in range(value):
            date = today - timedelta(days=i)
            dates.append(date.strftime('%Y%m%d'))
    
    return dates


def setup_directories(base_path: str, years: List[int]) -> None:
    """
    Create directory structure for storing downloaded files.
    
    Args:
        base_path: Base directory path
        years: List of years to create subdirectories for
    """
    os.makedirs(base_path, exist_ok=True)
    for year in years:
        year_path = os.path.join(base_path, str(year))
        os.makedirs(year_path, exist_ok=True)
        logger.info(f"Created directory: {year_path}")


def extract_pdf_links_shortcut(date_str: str) -> Tuple[List[Tuple[str, str]], bool]:
    """
    Extract PDF links using shortcut URL method.
    
    Args:
        date_str: Date in YYYYMMDD format
        
    Returns:
        Tuple of (links: List[Tuple[filename, url]], shortcut_worked: bool)
        shortcut_worked indicates if the shortcut URL successfully loaded the page
    """
    success, links = test_shortcut_url(date_str)
    return (links, success)


def extract_pdf_links_selenium(year: int, date_str: str) -> List[Tuple[str, str]]:
    """
    Extract PDF links using Selenium navigation (fallback method).
    
    Args:
        year: Year as integer
        date_str: Date in YYYYMMDD format (YYYYMMDD)
        
    Returns:
        List of (filename, url) tuples
    """
    driver = None
    try:
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument(f'user-agent={USER_AGENT}')
        
        # Initialize driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(30)
        
        # Navigate to main page
        logger.info(f"Navigating to main page for Selenium fallback")
        driver.get(BASE_URL_MAIN)
        
        # Wait for and select year dropdown
        wait = WebDriverWait(driver, 20)
        year_select_element = wait.until(
            EC.presence_of_element_located((By.NAME, "DiarioGrid"))
        )
        year_select = Select(year_select_element)
        year_select.select_by_value(str(year))
        
        # Wait for date dropdown to appear
        date_select_element = wait.until(
            EC.presence_of_element_located((By.NAME, "DiarioAjaxGridBaixar"))
        )
        date_select = Select(date_select_element)
        
        # Format date for selection (need to check format - might be DD/MM/YYYY or YYYYMMDD)
        # Try different formats
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        date_formats = [
            date_obj.strftime('%d/%m/%Y'),
            date_obj.strftime('%Y-%m-%d'),
            date_str
        ]
        
        date_selected = False
        for date_format in date_formats:
            try:
                date_select.select_by_value(date_format)
                date_selected = True
                break
            except:
                try:
                    date_select.select_by_visible_text(date_format)
                    date_selected = True
                    break
                except:
                    continue
        
        if not date_selected:
            logger.warning(f"Could not select date {date_str} in dropdown")
            return []
        
        # Wait for popup/table to appear
        # Look for table with links
        try:
            table = wait.until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
        except TimeoutException:
            logger.warning(f"No table found for date {date_str}")
            return []
        
        # Extract links from table
        links = []
        rows = table.find_elements(By.TAG_NAME, "tr")
        
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            for cell in cells:
                try:
                    link_elem = cell.find_element(By.TAG_NAME, "a")
                    href = link_elem.get_attribute('href')
                    if not href or not ('.pdf' in href.lower() or 'download' in href.lower() or 'baixar' in href.lower()):
                        continue
                    
                    # Try multiple methods to get filename
                    filename = None
                    
                    # Method 1: Extract from URL
                    filename = extract_filename_from_url(href)
                    
                    # Method 2: Look in table cells (often filename is in adjacent cell)
                    if not filename or filename.lower() in ['visualizar.pdf', 'visualizar', 'baixar.pdf', 'download.pdf']:
                        for other_cell in cells:
                            try:
                                cell_text = other_cell.text.strip()
                                # Skip if it's the link text or common action words
                                if (cell_text and 
                                    cell_text.lower() not in ['visualizar', 'baixar', 'download', 'ver'] and
                                    (cell_text.lower().endswith('.pdf') or len(cell_text) > 5)):
                                    filename = cell_text
                                    if not filename.endswith('.pdf'):
                                        filename += '.pdf'
                                    break
                            except:
                                continue
                    
                    # Method 3: Try to get from Content-Disposition header
                    if not filename or filename.lower() in ['visualizar.pdf', 'visualizar', 'baixar.pdf', 'download.pdf']:
                        filename = get_filename_from_content_disposition(href, {'User-Agent': USER_AGENT})
                    
                    # Method 4: Use link text if it's not a generic action word
                    if not filename or filename.lower() in ['visualizar.pdf', 'visualizar', 'baixar.pdf', 'download.pdf']:
                        link_text = link_elem.text.strip()
                        if link_text and link_text.lower() not in ['visualizar', 'baixar', 'download', 'ver']:
                            filename = link_text
                        else:
                            # Fallback: use basename from URL
                            filename = os.path.basename(urlparse(href).path) or 'documento.pdf'
                    
                    # Ensure .pdf extension
                    if not filename.endswith('.pdf'):
                        filename += '.pdf'
                    
                    # Clean filename
                    filename = unquote(filename)
                    
                    links.append((filename, href))
                except NoSuchElementException:
                    continue
                except Exception as e:
                    logger.debug(f"Error extracting link from cell: {e}")
                    continue
        
        return links
        
    except Exception as e:
        logger.error(f"Error in Selenium extraction for {date_str}: {e}")
        return []
    finally:
        if driver:
            driver.quit()


def should_overwrite_file(filepath: str, remote_size: Optional[int]) -> bool:
    """
    Check if file exists and prompt user to skip or overwrite.
    
    Args:
        filepath: Local file path
        remote_size: Remote file size in bytes (if available)
        
    Returns:
        True to overwrite, False to skip
    """
    if not os.path.exists(filepath):
        return True
    
    local_size = os.path.getsize(filepath)
    
    if remote_size is None:
        prompt = f"File exists: {os.path.basename(filepath)} (Local: {local_size:,} bytes, Remote size unknown). (s)kip or (o)verwrite? "
    else:
        prompt = f"File exists: {os.path.basename(filepath)} (Local: {local_size:,} bytes, Remote: {remote_size:,} bytes). (s)kip or (o)verwrite? "
    
    while True:
        try:
            choice = input(prompt).strip().lower()
            if choice == 's' or choice == 'skip':
                return False
            elif choice == 'o' or choice == 'overwrite':
                return True
            else:
                print("Please enter 's' to skip or 'o' to overwrite")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(0)


def download_file(url: str, filepath: str, max_retries: int = 3) -> Tuple[bool, str]:
    """
    Download file with progress bar and retry logic.
    Also tries to get actual filename from Content-Disposition header.
    
    Args:
        url: URL to download
        filepath: Local file path to save to
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (success: bool, status: str) where status is 'downloaded', 'skipped', or 'failed'
    """
    headers = {'User-Agent': USER_AGENT}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            
            # Try to get actual filename from Content-Disposition header
            content_disposition = response.headers.get('Content-Disposition', '')
            actual_filename = None
            if content_disposition:
                filename_match = re.search(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', content_disposition, re.IGNORECASE)
                if filename_match:
                    actual_filename = filename_match.group(1).strip('"\'')
                    actual_filename = unquote(actual_filename)
            
            # If we got a better filename from headers, update filepath
            if actual_filename and actual_filename.lower().endswith('.pdf'):
                # Only update if current filename is generic
                current_basename = os.path.basename(filepath).lower()
                if current_basename in ['visualizar.pdf', 'baixar.pdf', 'download.pdf', 'documento.pdf']:
                    # Update filepath with actual filename
                    dir_path = os.path.dirname(filepath)
                    filepath = os.path.join(dir_path, actual_filename)
                    logger.info(f"Using filename from Content-Disposition: {actual_filename}")
            
            # Get file size from headers
            total_size = int(response.headers.get('content-length', 0))
            
            # Get filename for display
            filename = os.path.basename(filepath)
            
            # Check if should overwrite
            if os.path.exists(filepath):
                if not should_overwrite_file(filepath, total_size):
                    logger.info(f"Skipping {filename}")
                    return (True, 'skipped')
            
            # Download with progress bar
            with open(filepath, 'wb') as f:
                if total_size > 0:
                    with tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=filename,
                        ncols=100
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # Unknown size, still show progress
                    with tqdm(
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=filename,
                        ncols=100
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            
            logger.info(f"Successfully downloaded: {filename}")
            return (True, 'downloaded')
            
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}. Retrying...")
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts: {e}")
                return (False, 'failed')
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            if attempt < max_retries - 1:
                logger.warning(f"Retrying...")
            else:
                return (False, 'failed')
    
    return (False, 'failed')


def main():
    """Main execution function."""
    logger.info("Starting Diários Oficiais Ceará Scraper")
    
    # Get user input
    mode, value = get_user_input()
    logger.info(f"Mode: {mode}, Value: {value}")
    
    # Generate dates
    dates = generate_dates(mode, value)
    logger.info(f"Generated {len(dates)} dates to process")
    
    # Extract unique years from dates
    years = sorted(set(int(date[:4]) for date in dates))
    
    # Setup directories
    setup_directories(BASE_DIR, years)
    
    # Statistics
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    dates_with_publications = 0
    dates_without_publications = 0
    
    # Process each date
    for date_str in dates:
        year = int(date_str[:4])
        logger.info(f"Processing date: {date_str}")
        
        # Try shortcut method first
        links, shortcut_worked = extract_pdf_links_shortcut(date_str)
        
        # If shortcut failed to load page, try Selenium
        # If shortcut worked but returned no links, it means no publications (don't try Selenium)
        if not shortcut_worked:
            logger.info(f"Shortcut method failed for {date_str}, trying Selenium fallback")
            links = extract_pdf_links_selenium(year, date_str)
        
        if not links:
            logger.info(f"No publications found for date {date_str}")
            dates_without_publications += 1
            continue
        
        dates_with_publications += 1
        logger.info(f"Found {len(links)} files for date {date_str}")
        
        # Download each file
        for filename, url in links:
            # Clean filename (remove invalid characters)
            safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
            if not safe_filename.endswith('.pdf'):
                safe_filename += '.pdf'
            
            filepath = os.path.join(BASE_DIR, str(year), safe_filename)
            
            success, status = download_file(url, filepath)
            if status == 'downloaded':
                total_downloaded += 1
            elif status == 'skipped':
                total_skipped += 1
            elif status == 'failed':
                total_failed += 1
    
    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Total dates processed: {len(dates)}")
    print(f"Dates with publications: {dates_with_publications}")
    print(f"Dates without publications: {dates_without_publications}")
    print(f"Files downloaded: {total_downloaded}")
    print(f"Files skipped: {total_skipped}")
    print(f"Files failed: {total_failed}")
    print("="*60)
    logger.info("Scraper finished")


if __name__ == "__main__":
    main()

