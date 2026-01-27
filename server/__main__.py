"""
Module entrypoint so you can run the backend with:

    python -m server
"""

import uvicorn


def main() -> None:
    uvicorn.run("server.main:app", host="0.0.0.0", port=5050)


if __name__ == "__main__":
    main()

