import subprocess


def convert_pdf_to_html(pdf_path, html_path):
    try:
        command = f"pdf2htmlEX {pdf_path} --dest-dir {html_path}"
        subprocess.call(command, shell=True)
    except FileNotFoundError:
        print('pdf2htmlEX is not installed or not found')
    except subprocess.CalledProcessError as e:
        print(f'Conversion failed with exit code {e.returncode}')

    