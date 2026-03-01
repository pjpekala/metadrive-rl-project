try:
    import dm_tree as tree
    print(f"Successfully imported dm_tree as tree from: {tree.__file__}")
except ImportError:
    print("Failed to import dm_tree as tree.")
