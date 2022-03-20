from gluonts.dataset.common import ListDataset


def convert_to_list_dataset(nav_table):
    """
    Convert single nav_table (DataFrame) to GluonTS ListDataset.

    Args:
        - nav_table: the pandas table.
    Returns:
        - dataset: the ListDataset.
    """
    dataset = ListDataset([
        {"start": nav_table.index[0],
         "target": nav_table.value}], freq="D")
    return dataset
