import pandas as pd
import numpy as np
import re
import warnings
from typing import Tuple, List, Dict, List, Optional, Union
from rapidfuzz import process, fuzz

class DataCleaner:
    """
    A class for cleaning datasets for machine learning models.
    Contains methods to handle various cleaning operations with options for
    handling or removing NaN values based on model requirements.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialises the DataCleaner with a DataFrame.
        Args:
            df (pd.DataFrame): The initial DataFrame to clean.
        """
        self._original_df = df.copy()  # A pristine, un-touchable copy
        self.current_df = df.copy()    # The "live" DataFrame that will be modified
        
        self.TARGET_COLUMN = "ALLOCATED DIVISION"
        
        self.COLS_TO_KEEP = ['PBS NODE', 'ELEMENTARY SYSTEM', 'ECS CODE', #'DESCRIPTION',
               'BUILDING CODE', 'CONSUMER ROOM', 'REQUIRED ELECTRICAL DIV', 'ELECTRICAL LOAD TYPE',
            #    'REQUIRED VOLTAGE', 'PN - RATED POWER [kW]',
               'REQUIRED TYPE OF VOLTAGE', 'REQUIRED POWER', 'BELONGS TO PACKAGE',
               'PACKAGE ECS CODE', 'HEATER RESISTOR FOR MOTOR', 'INTERLOCKED LOADS',
               'MULTIPLE POWER SUPPLY', 'MAINTENANCE STATE A', 'MAINTENANCE STATE B',
               'MAINTENANCE STATE C', 'MAINTENANCE STATE D', 'MAINTENANCE STATE E',
               'MAINTENANCE STATE F', 'OPERATED BY MOBILE SOURCE', 'I&C ORDER',
               'MOTOR BEHAVIOR ON VOLTAGE LOSS', 'OPERATED WINTER', 'OPERATED SUMMER', #'ADDITIONAL REQUIREMENTS',
               'FUNC DESIGN REPORT', 'ELECTRICAL DATA MATURITY', 'ORDER PRIORITY MODULE NEEDED',
               'CONTAINMENT ISOLATION VALVES', 'SAFETY CLASS', 'ELECTRICAL REQ',
               'SEISMIC REQUIREMENT DBH', 'SEISMIC REQUIREMENT DEH',
               'EARTHQUAKES OPERABILITY', 'OPERATION NORMAL SUPPLY',
               'OPERATION AUXILIARY SUPPLY', 'ELEC BACK UP LOOP', 'ELEC BACK UP SBO',
               'ELEC BACK UP SA', 'UNINTERRUPTED POWER SUPPLY',
            #    'EFFICIENCY AT PN [%]', 'POWER FACTOR (COS PHI) AT PN', 'EFFICIENCY AT PU [%]', 'POWER FACTOR (COS PHI) AT PU',  ## violet blue
               'ALLOCATED DIVISION', 'POWER SOURCE ECS CODE', # 'ALLOCATED VOLTAGE', 'POWER SOURCE BUILDING CODE',  ## last 2 is defined at the same time as target
               'SWITCHING DEVICE ECS CODE', 'SWITCHING DEVICE TYPE']
        
        self.EXPECTED_VALUES = {
            "REQUIRED ELECTRICAL DIV" : [
                "unset", "DIV1", "DIV2", "DIV3", "DIV4", "DIV9", "DIV1 or DIV2", "DIV1 or DIV3", "DIV2 or DIV3", "DIV1 or DIV2 or DIV3", "NA", "LATER"
                ],
            "ELECTRICAL LOAD TYPE" : [
                "unset", "CONTROL VALVE", "ON-OFF VALVE", "MOTOR 1 DIRECTION 1 SPEED", "MOTOR 1 DIRECTION 2 SPEEDS", "MOTOR 2 DIRECTIONS",
                "INTELLIGENT CONTROL VALVE", "INTELLIGENT ON-OFF VALVE", "SOLENOID VALVE", "HEATER FOR MOTOR", "HEATER FOR PROCESS", "HEATER FOR HVAC",
                "ELECTRICAL CUBICLE", "BUILDING BOARD", "FINAL BOARD", "ELECTRICAL BOX", "SPEED-VARIATOR", "POWER-CONVERTER", "RECTIFIER", "INVERTER",
                "INCOMING", "OUTGOING", "SWITCHBOARD AUXILIARY", "TRANSFORMER", "BATTERY", "MOBILE", "NA", "LATER", "BLACK BOX", "SUBDISTRIBUTION MASTER",
                "SUBDISTRIBUTION SLAVE", "MOTOR", "HEATER", "SUBDISTRIBUTION"
                ],
            "REQUIRED VOLTAGE": ["unset", "10000", "690", "400", "230", "220", "125", "48", "24", "NA", "LATER"],
            "REQUIRED TYPE OF VOLTAGE": ["unset", "AC1", "AC2", "AC3", "AC3N", "DC", "NA", "LATER"],
            "BELONGS TO PACKAGE": ["unset", "YES", "NO", "LATER", "NA"],
            "HEATER RESISTOR FOR MOTOR": ["unset", "YES", "NO", "LATER", "NA"],
            "MULTIPLE POWER SUPPLY": ["unset", "YES", "NO", "LATER", "NA"],
            "MAINTENANCE STATE A": ["unset", "YES", "NO", "LATER", "NA", "CONDITIONAL"],
            "MAINTENANCE STATE B": ["unset", "YES", "NO", "LATER", "NA", "CONDITIONAL"],
            "MAINTENANCE STATE C": ["unset", "YES", "NO", "LATER", "NA", "CONDITIONAL"],
            "MAINTENANCE STATE D": ["unset", "YES", "NO", "LATER", "NA", "CONDITIONAL"],
            "MAINTENANCE STATE E": ["unset", "YES", "NO", "LATER", "NA", "CONDITIONAL"],
            "MAINTENANCE STATE F": ["unset", "YES", "NO", "LATER", "NA", "CONDITIONAL"],
            "I&C ORDER": ["unset", "CENTRALIZED I&C", "DEDICATED I&C", "CENTRALIZED I&C AND DEDICATED I&C", "NO", "LATER", "NA", "CENTRALIZED I&C AND PRIORITIZED DEDICATED I&C"],
            "MOTOR BEHAVIOR ON VOLTAGE LOSS": ["CASE 1", "CASE 2", "CASE 2a", "CASE 3", "CASE 4", "unset", "NA", "LATER"],
            "OPERATED WINTER": ["unset", "0", "1", "PERMANENT", "INTERMITTENT", "RARE", "NONE", "LATER", "NA"],
            "OPERATED SUMMER": ["unset", "0", "1", "PERMANENT", "INTERMITTENT", "RARE", "NONE", "LATER", "NA"],
            "ELECTRICAL DATA MATURITY": ["0", "1", "2", "3", "4", "5", "unset", "NA", "LATER"],
            "ORDER PRIORITY MODULE NEEDED": ["unset", "YES", "NO", "LATER", "NA"],
            "CONTAINMENT ISOLATION VALVES": ["ICIV", "OCIV", "ICIV-like", "OCIV-like", "NO", "LATER", "unset", "NA",],
            "SAFETY CLASS": ["S1", "S2", "S3", "NC"],
            "ELECTRICAL REQ": ["C1", "C2", "C3", "NR"],
            "SEISMIC REQUIREMENT DBH": ["O-OPERABILITY", "F-FUNCTIONAL CAPABILITY", "I-INTEGRITY", "S-STABILITY"],
            "SEISMIC REQUIREMENT DEH": ["O-OPERABILITY", "F-FUNCTIONAL CAPABILITY", "I-INTEGRITY", "S-STABILITY"],
            "EARTHQUAKES OPERABILITY": ["unset", "YES", "NO", "LATER", "NA"],
            "OPERATION NORMAL SUPPLY": ["YES", "PT", "LATER", "NO", "unset", "NA"],
            "OPERATION AUXILIARY SUPPLY": ["unset", "YES", "NO", "LATER", "NA"],
            "ELEC BACK UP LOOP": ["YES FOR SAFETY", "YES FOR PLANT AVAILABILITY", "NO", "LATER", "NA", "unset"],
            "ELEC BACK UP SBO": ["YES", "NO", "LATER", "NA", "unset"],
            "ELEC BACK UP SA": ["YES", "NO", "LATER", "NA", "unset"],
            "UNINTERRUPTED POWER SUPPLY": ["YES", "MANUAL RESUPPLY", "NO", "LATER", "NA", "unset"],
            "ALLOCATED DIVISION": ["DIV1", "DIV2", "DIV3", "DIV4", "DIV9", "LATER", "NA", "unset"],
            "ALLOCATED VOLTAGE": ["unset", "10000", "690", "400", "230", "220", "125", "48", "24", "NA", "LATER"],
            "SWITCHING DEVICE TYPE": ["unset", "CB", "FC", "FSD", "SD", "SD+DIODE", "NA", "LATER"]
        }

        self.BOOLEAN_COLUMNS = [
            'BELONGS TO PACKAGE',
            'HEATER RESISTOR FOR MOTOR',
            'MULTIPLE POWER SUPPLY',
            'MAINTENANCE STATE A',
            'MAINTENANCE STATE B',
            'MAINTENANCE STATE C',
            'MAINTENANCE STATE D',
            'MAINTENANCE STATE E',
            'MAINTENANCE STATE F',
            'OPERATED BY MOBILE SOURCE',
            'OPERATED WINTER',
            'OPERATED SUMMER',
            'ORDER PRIORITY MODULE NEEDED',
            'EARTHQUAKES OPERABILITY',
            'OPERATION AUXILIARY SUPPLY',
            'ELEC BACK UP SBO',
            'ELEC BACK UP SA'
        ]

        # Floats
        self.NUMERIC_COLUMNS = [
            'REQUIRED POWER',
            # 'PN - RATED POWER [kW]',
            # 'EFFICIENCY AT PN [%]',
            # 'POWER FACTOR (COS PHI) AT PN',
            # 'EFFICIENCY AT PU [%]',
            # 'POWER FACTOR (COS PHI) AT PU'
        ]


    @property
    def original_df(self) -> pd.DataFrame:
        """Returns the original, unmodified DataFrame."""
        return self._original_df

    @property
    def df(self) -> pd.DataFrame:
        """Returns the current state of the DataFrame."""
        return self.current_df
    
    @property
    def target(self) -> pd.DataFrame:
        """Returns the current state of the DataFrame."""
        return self.TARGET_COLUMN
    
    @property
    def numeric_cols(self) -> pd.DataFrame:
        """Returns the current state of the DataFrame."""
        return self.NUMERIC_COLUMNS

    # ---- 1. Find and clean unprintable/whitespace column names ---    
    def find_unprintable_columns(self):
        """Find columns with unprintable characters or whitespace issues in names"""
        issues = []

        for col in self.current_df.columns:
            problems = []
            # Check for unprintable characters
            if not col.isprintable():
                problems.append("unprintable characters")
            # Check for leading/trailing whitespace
            if col != col.strip():
                problems.append("leading/trailing whitespace")
            # Check for internal multiple spaces
            if '  ' in col:
                problems.append("multiple internal spaces")
            # Check for tabs
            if '\t' in col:
                problems.append("tab characters")
            # Check for newlines
            if '\n' in col or '\r' in col:
                problems.append("newline characters")
            if problems:
                issues.append({
                    'column': repr(col),
                    'issues': ', '.join(problems),
                    'cleaned_version': self._clean_col_name(col)
                })

        if issues:
            print(f"Found {len(issues)} columns with issues:")
            for item in issues:
                print(f"  Column: {item['column']}")
                print(f"    Issues: {item['issues']}")
                print(f"    Will become: '{item['cleaned_version']}'")
                print()
        else:
            print("✓ All column names are clean!")

        return issues

    def standardise_column_names(self):
        """Standardise column names by removing control characters like \\n, \\t"""
        old_cols = self.current_df.columns.tolist()
        self.current_df.columns = [self._clean_col_name(col) for col in self.current_df.columns]
        changed = sum(1 for old, new in zip(old_cols, self.current_df.columns) if old != new)
        print(f"✓ Cleaned {changed} column names")
        return self.current_df
    
    @staticmethod
    def _clean_col_name(col: str) -> str:
        """Internal helper to clean a single column name."""
        cleaned_col = "".join(char if char.isprintable() else ' ' for char in col)
        cleaned_col = re.sub(r'\s+', ' ', cleaned_col)
        return cleaned_col.strip()
    
    
    # ---- 2. Drop columns that are not of interest ---
    def drop_columns(self):
        """Drop columns that are of no interest"""
        self.current_df = self.current_df[self.COLS_TO_KEEP]
        return self.current_df


    # ---- 3. Clean the target column ---
    def clean_target(self, only_later):
        """Turn empties into LATER"""
        if only_later:
            self.current_df[self.TARGET_COLUMN] = self.current_df[self.TARGET_COLUMN].fillna("LATER").replace(["building", "unset"], "LATER")
        else:
            self.current_df[self.TARGET_COLUMN] = self.current_df[self.TARGET_COLUMN].fillna("unset").replace("building", "LATER")
        return self.current_df
    
    
    # ---- 4. Standardise NA values ---
    def NA_values(self):
        """
        Standardize all variations of N/A to 'NA' string across all columns.
        """
        # List of N/A variations to replace
        na_variations = [
            'N/A', 'n/a', 'N/a', 'n/A',
            'na', 'Na', 'nA', 'NA',
            'n.a.', 'N.A.', 'n.a', 'N.A',
            'not available', 'Not Available',
            'not applicable', 'Not Applicable'
        ]

        # Replace all variations with 'NA'
        for col in self.current_df.columns:
            for na_var in na_variations:
                # Case-insensitive replacement
                mask = self.current_df[col].astype(str).str.lower() == na_var.lower()
                self.current_df.loc[mask, col] = 'NA'

        print("✓ Standardized all N/A variations to 'NA'")
        return self.current_df
    

    # ---- 5. Find and clean whitespace ---
    def find_whitespace_in_values(self):
        """Find columns with leading/trailing whitespace in values"""
        whitespace_info = []
        string_cols = self.current_df.select_dtypes(include=['object', 'string']).columns
        for col in string_cols:
            has_whitespace = self.current_df[col].astype(str).str.strip() != self.current_df[col].astype(str)
            if has_whitespace.any():
                count = has_whitespace.sum()
                whitespace_mask = has_whitespace
                examples_original = self.current_df[col][whitespace_mask].head(3).tolist()
                examples_cleaned = [str(val).strip() for val in examples_original]
                examples_orig_str = ' | '.join([f'"{val}"' for val in examples_original])
                examples_clean_str = ' | '.join([f'"{val}"' for val in examples_cleaned])
                whitespace_info.append({
                    'column': col,
                    'affected_rows': count,
                    'percentage': round(count / len(self.current_df) * 100, 2),
                    'examples_before': examples_orig_str,
                    'examples_after': examples_clean_str
                })
        return pd.DataFrame(whitespace_info)

    def trim_whitespace(self, columns=None):
        """Trim whitespace from specified columns (or all string columns if None)"""
        string_cols = self.current_df.select_dtypes(include=['object', 'string']).columns
        if columns is not None:
            string_cols = [col for col in columns if col in string_cols]
        
        for col in string_cols:
            self.current_df[col] = self.current_df[col].str.strip()
        
        print(f"✓ Trimmed whitespace from {len(string_cols)} columns")
        return self.current_df
    

    # ---- 6. Find and clean case insensitive duplicates ---
    def find_case_insensitive_duplicates(self):
        """
        Finds columns with case-insensitive duplicates (e.g., 'Apple', 'apple').
        Returns a DataFrame summarizing the issues for easy display.
        """
        results = []
        string_cols = self.current_df.select_dtypes(include=['object', 'string']).columns
        for col in string_cols:
            series = self.current_df[col]
            clean_series = series.dropna().astype(str)
            if len(clean_series) == 0:
                continue
            case_map = {}
            for value in clean_series.unique():
                lower_val = value.lower()
                if lower_val in case_map:
                    case_map[lower_val].append(value)
                else:
                    case_map[lower_val] = [value]
            duplicate_groups = [group for group in case_map.values() if len(group) > 1]
            if duplicate_groups:
                value_counts = self.current_df[col].value_counts()
                total_affected_rows = 0
                example_groups = []
                for group in duplicate_groups:
                    total_affected_rows += value_counts[group].sum()
                    most_frequent_form = max(group, key=lambda x: value_counts.get(x, 0))
                    group_str = ' | '.join([f'"{val}"' for val in sorted(group)])
                    example_groups.append(f'{group_str} -> "{most_frequent_form}"')
                results.append({
                    'column': col,
                    'duplicate_groups': len(duplicate_groups),
                    'affected_rows': total_affected_rows,
                    'examples': ' || '.join(example_groups[:3])
                })
        return pd.DataFrame(results)

    def standardise_case(self, columns: list):
        """
        Standardises the casing of values in the specified columns.
        """
        standardised_count = 0
        for col in columns:
            if col not in self.current_df.columns:
                continue
            series = self.current_df[col]
            clean_series = series.dropna().astype(str)
            if len(clean_series) == 0:
                continue
            case_map = {}
            for value in clean_series.unique():
                lower_val = value.lower()
                if lower_val in case_map:
                    case_map[lower_val].append(value)
                else:
                    case_map[lower_val] = [value]
            duplicate_groups = [group for group in case_map.values() if len(group) > 1]
            if not duplicate_groups:
                continue
            value_counts = self.current_df[col].value_counts()
            replacement_map = {}
            for group in duplicate_groups:
                most_frequent_form = max(group, key=lambda x: value_counts.get(x, 0))
                for variant in group:
                    if variant != most_frequent_form:
                        replacement_map[variant] = most_frequent_form
            if replacement_map:
                self.current_df[col] = self.current_df[col].replace(replacement_map)
                standardised_count += 1
        
        print(f"✓ Standardised case in {standardised_count} columns")
        return self.current_df


    # ---- 7. Fuzzy String Matching ----
    @staticmethod
    def _normalise_for_comparison(s: str) -> str:
        """Intelligently cleans a string for a base similarity comparison."""
        if not isinstance(s, str):
            return ""
        s_lower = s.lower()
        s_lower = re.sub(r'tbc\s*\(proposition\s*-?|local\s*|à\s*confirmer|pp\s*\d', '', s_lower)
        s_lower = re.sub(r'[\s-]+', '', s_lower)
        s_lower = s_lower.strip("()[]{}'\"- ")
        return s_lower

    def find_fuzzy_duplicates(self, threshold: int = 85, min_length: int = 3):
        """
        Finds groups of similar strings (potential typos) in categorical columns.
        """
        issue_list = []
        string_cols = self.current_df.select_dtypes(include=['object', 'string']).columns
        for col in string_cols:
            series = self.current_df[col]
            if series.nunique() < 2 or series.nunique() > 2000:
                continue
            categories = series.dropna().unique().tolist()
            filtered_cats = [
                cat for cat in set(categories)
                if isinstance(cat, str) and len(cat) >= min_length and not re.search(r'\d', cat)
            ]
            if len(filtered_cats) < 2:
                continue
            normalised_cats = [self._normalise_for_comparison(cat) for cat in filtered_cats]
            score_matrix = process.cdist(normalised_cats, normalised_cats, scorer=fuzz.ratio, score_cutoff=threshold)
            groups = []
            processed_indices = set()
            for i in range(len(filtered_cats)):
                if i in processed_indices:
                    continue
                nonzero_result = score_matrix[i].nonzero()
                if isinstance(nonzero_result, tuple) and len(nonzero_result) > 0:
                    similar_indices = nonzero_result[0] if len(nonzero_result) == 1 else nonzero_result[1]
                else:
                    continue
                if len(similar_indices) > 1:
                    current_group = {filtered_cats[j] for j in similar_indices}
                    groups.append(sorted(list(current_group)))
                    processed_indices.update(similar_indices)
            if groups:
                issue_list.append({'column': col, 'fuzzy_groups': groups})
        return issue_list

    def standardise_fuzzy_values(self, column: str, mappings: dict):
        """
        Standardises values in a column based on a provided mapping.
        """
        if column in self.current_df.columns and mappings:
            self.current_df[column] = self.current_df[column].replace(mappings)
            print(f"✓ Applied {len(mappings)} fuzzy mappings to column '{column}'")
        return self.current_df


    # ---- 8. Type Conversion Methods ----
    def convert_to_numeric(self):
        """Converts specified columns to numeric type"""
        columns = self.NUMERIC_COLUMNS
        converted = []
        for col in columns:
            if col not in self.current_df.columns:
                print(f"⚠ Column '{col}' not found, skipping")
                continue
            
            try:
                self.current_df[col] = pd.to_numeric(
                    self.current_df[col].astype(str).str.replace(',', '.', regex=False), 
                    errors='coerce'
                )
                converted.append(col)
            except Exception as e:
                print(f"⚠ Error converting '{col}': {e}")
        
        print(f"✓ Converted {len(converted)} columns to numeric type")
        return self.current_df
    
    
    # ---- 9. Handle unexpected and missing values ----
    # 9.1 Replace errors with later or unset
    def replace_erreur_values(self, fill_value="unset"):
        """
        Replace all values containing 'erreur' with 'later' or 'unset' across the entire DataFrame.
        Shows which columns were affected and how many values in each.
        """
        total_count = 0
        affected_columns = {}

        for col in self.current_df.columns:
            if self.current_df[col].dtype == 'object':
                mask = self.current_df[col].astype(str).str.contains('erreur', case=False, na=False)
                count = mask.sum()

                if count > 0:
                    affected_columns[col] = count
                    total_count += count
                    self.current_df.loc[mask, col] = fill_value

        print(f"✓ Replaced {total_count} values containing 'erreur' with '{fill_value}'")
        print(f"  Affected {len(affected_columns)} column(s):")

        for col, count in affected_columns.items():
            print(f"    - {col}: {count} value(s)")

        return self.current_df
    
    # 9.2 Clean consumer rooms
    def validate_long_code_format(self, column_name: str):
        """
        Validate that a column contains long codes in the correct format.
        Expected format: [-0-9]ABC1234DE[-A-Z]
        - First char: - or digit (0-9)
        - Middle: 3 letters + 4 digits + 2 letters
        - Last char: - or letter

        Groups unexpected values and shows count + sample indices.

        Returns:
            DataFrame with unexpected values grouped by unique value
        """
        import re

        if column_name not in self.current_df.columns:
            print(f"⚠ Column '{column_name}' not found")
            return pd.DataFrame()

        # Multiple valid patterns
        patterns = [
            r'^[-0-9][A-Z]{3}\d{4}[A-Z]{2}[-A-Z]$',      # -ABC1234DE-
            r'^[-0-9][A-Z]{3}\d{4}[A-Z]{2}[0-9]$',      # -ABC1234DE-
            r'^[-0-9][A-Z]{3}\d{4}[A-Z]{3}$',            # -ABC1234DEF
            r'^[A-Z]{3}\d{4}[A-Z]{2}$',                  # ABC0221KL
            r'^[-0-9][A-Z]{3}\d{1}[A-Z]{1}\d{2}[A-Z]{2}[-A-Z]$',  # -HLL0A14BB-
        ]

        unexpected_values = {}

        for idx, val in self.current_df[column_name].items():
            if pd.isna(val):
                continue
            
            val_str = str(val).strip()

            # Check if it matches any valid pattern
            is_valid = any(re.match(pattern, val_str) for pattern in patterns)

            if not is_valid:
                if val_str not in unexpected_values:
                    unexpected_values[val_str] = []
                unexpected_values[val_str].append(idx)

        if unexpected_values:
            # Create summary DataFrame
            summary = []
            for value, indices in unexpected_values.items():
                length = len(value)

                # Categorize
                if length <= 4:
                    category = "short code"
                elif length < 9:
                    category = "too short for long code"
                elif length > 11:
                    category = "too long"
                else:
                    category = "invalid format"

                # Get sample indices (first 5)
                sample_indices = indices[:5]

                summary.append({
                    'value': value,
                    'count': len(indices),
                    'length': length,
                    'category': category,
                    'sample_indices': sample_indices
                })

            unexpected_df = pd.DataFrame(summary).sort_values('count', ascending=False)

            print(f"\n{'='*60}")
            print(f"VALIDATION: {column_name}")
            print(f"{'='*60}")
            print(f"Total rows with unexpected values: {sum(len(indices) for indices in unexpected_values.values())}")
            print(f"Unique unexpected values: {len(unexpected_values)}")
            print(f"\nBreakdown by category:")

            category_counts = unexpected_df.groupby('category')['count'].sum()
            for cat, count in category_counts.items():
                print(f"  {cat}: {count} rows")

            print(f"\nUnexpected values (sorted by frequency):")
            print(unexpected_df.to_string(index=False))
            print(f"{'='*60}\n")

            return unexpected_df
        else:
            print(f"✓ Column '{column_name}': All values are in correct format")
            return pd.DataFrame()
    
    # 9.2.2 Clean consumer rooms
    def consumer_rooms(self, fill_value="unset"):
        """
        Clean the CONSUMER ROOM column by standardising various invalid values.
        Values containing 'TBD'/'TBC' or other 'later' indicators → 'LATER'
        Other invalid values → 'later' or 'unset'
        """
        column_name = 'CONSUMER ROOM'

        if column_name not in self.current_df.columns:
            print(f"⚠ Column '{column_name}' not found")
            return self.current_df

        # Patterns that mean "to be defined later"
        later_patterns = ['tbc', 'tbd', 'a definir', 'à définir', 'à confimer', 'a confimer', '?']

        # Values that should be later (exact matches)
        exact_values = ['0', 'non trouvé', 'pas en maquette', 'supprimé', '-']

        later_count = 0
        unset_count = 0

        for idx, val in self.current_df[column_name].items():
            if pd.isna(val):
                continue

            val_str = str(val).strip()
            val_lower = val_str.lower()

            # Check if any "later" pattern appears in the string
            if any(pattern in val_lower for pattern in later_patterns):
                self.current_df.at[idx, column_name] = 'LATER'
                later_count += 1
            # Check for exact later values (case-insensitive)
            elif val_str in exact_values or val_str.lower() in [v.lower() for v in exact_values]:
                self.current_df.at[idx, column_name] = fill_value
                if fill_value == "LATER":
                    later_count += 1
                else:
                    unset_count += 1

        print(f"✓ Cleaned 'CONSUMER ROOM' column:")
        print(f"  {later_count} values → 'LATER'")
        if unset_count > 0:
            print(f"  {unset_count} values → 'unset'")

        return self.current_df
    
    # 9.2.3 Clean package ecs codes
    def package_codes(self, fill_value="unset"):
        """
        Clean the package ecs code column by standardising various invalid values.
        Values containing 'TBD'/'TBC' or other 'later' indicators → 'LATER'
        Other invalid values → 'later' or 'unset'
        """
        column_name = 'PACKAGE ECS CODE'
        print(f"Column name: {column_name}")
        if column_name not in self.current_df.columns:
            print(f"⚠ Column '{column_name}' not found")
            return self.current_df

        # Patterns that mean "to be defined later"
        later_patterns = ['a définir', 'à définir', 'a definir', 'à definir', '?']

        # Values that should be later (exact matches)
        exact_values = ['-', 'NO']
        
        later_count = 0
        unset_count = 0

        for idx, val in self.current_df[column_name].items():
            if pd.isna(val):
                continue

            val_str = str(val).strip()
            val_lower = val_str.lower()

            # Check if any "later" pattern appears in the string
            if any(pattern in val_lower for pattern in later_patterns):
                self.current_df.at[idx, column_name] = 'LATER'
                later_count += 1
            # Check for exact later values (case-insensitive)
            elif val_str in exact_values or val_str.lower() in [v.lower() for v in exact_values]:
                self.current_df.at[idx, column_name] = fill_value
                if fill_value == "LATER":
                    later_count += 1
                else:
                    unset_count += 1

        print(f"✓ Cleaned '{column_name}' column:")
        print(f"  {later_count} values → 'LATER'")
        if unset_count > 0:
            print(f"  {unset_count} values → 'unset'")

        return self.current_df
    
    # 9.2.4 Clean interlocked loads
    def interlocked_loads(self, fill_value="unset"):
        """
        Clean the INTERLOCKED LOADS column by standardising various invalid values.
        Values containing 'TBD'/'TBC' or other 'later' indicators → 'LATER'
        Other invalid values → 'later' or 'unset'
        """
        column_name = 'INTERLOCKED LOADS'
        print(f"Column name: {column_name}")
        if column_name not in self.current_df.columns:
            print(f"⚠ Column '{column_name}' not found")
            return self.current_df

        # Patterns that mean "to be defined later"
        later_patterns = ['a définir', 'à définir', 'a definir', 'à definir', '?']

        # Values that should be later (exact matches)
        exact_values = ['-', 'NO']
        
        later_count = 0
        unset_count = 0

        for idx, val in self.current_df[column_name].items():
            if pd.isna(val):
                continue

            val_str = str(val).strip()
            val_lower = val_str.lower()

            # Check if any "later" pattern appears in the string
            if any(pattern in val_lower for pattern in later_patterns):
                self.current_df.at[idx, column_name] = 'LATER'
                later_count += 1
            # Check for exact later values (case-insensitive)
            elif val_str in exact_values or val_str.lower() in [v.lower() for v in exact_values]:
                self.current_df.at[idx, column_name] = fill_value
                if fill_value == "LATER":
                    later_count += 1
                else:
                    unset_count += 1

        print(f"✓ Cleaned '{column_name}' column:")
        print(f"  {later_count} values → 'LATER'")
        if unset_count > 0:
            print(f"  {unset_count} values → 'unset'")

        return self.current_df
    
    # 9.2.5 Clean switching device
    def switching_device(self, fill_value="unset"):
        """
        Clean the SWITCHING DEVICE ECS CODE column by standardising various invalid values.
        Values containing 'TBD'/'TBC' or other 'later' indicators → 'LATER'
        Other invalid values → 'later' or 'unset'
        """
        column_name = 'SWITCHING DEVICE ECS CODE'
        print(f"\nColumn name: {column_name}")
        if column_name not in self.current_df.columns:
            print(f"⚠ Column '{column_name}' not found")
            return self.current_df

        # Patterns that mean "to be defined later"
        later_patterns = ['a définir', 'à définir', 'a definir', 'à definir']

        # Values that should be later (exact matches)
        exact_values = ['NO', '-']
        
        later_count = 0
        unset_count = 0

        for idx, val in self.current_df[column_name].items():
            if pd.isna(val):
                continue

            val_str = str(val).strip()
            val_lower = val_str.lower()

            # Check if any "later" pattern appears in the string
            if any(pattern in val_lower for pattern in later_patterns):
                self.current_df.at[idx, column_name] = 'LATER'
                later_count += 1
            # Check for exact later values (case-insensitive)
            elif val_str in exact_values or val_str.lower() in [v.lower() for v in exact_values]:
                self.current_df.at[idx, column_name] = fill_value
                if fill_value == "LATER":
                    later_count += 1
                else:
                    unset_count += 1

        print(f"✓ Cleaned '{column_name}' column:")
        print(f"  {later_count} values → 'LATER'")
        if unset_count > 0:
            print(f"  {unset_count} values → 'unset'")

        return self.current_df
    
    # 9.3.1 Check for unexpected short codes
    def validate_short_code_format(self, column_name: str):
        """
        Validate that a column contains codes in the correct format.
        Expected formats: -ABC, 0-9ABC, or optionally ABC (3 uppercase letters)

        Groups unexpected values and shows count + sample indices.

        Args:
            column_name: Name of the column to validate
        Returns:
            DataFrame with unexpected values grouped by unique value
        """        
        if column_name not in self.current_df.columns:
            print(f"⚠ Column '{column_name}' not found")
            return pd.DataFrame()

        # Accepts: -ABC, 0-9ABC, or ABC
        pattern = r'^[-0-9][A-Z]{3}$|^[A-Z]{3}$'

        unexpected_values = {}

        for idx, val in self.current_df[column_name].items():
            if pd.isna(val):
                continue
            
            val_str = str(val).strip()

            # Check if it matches the pattern
            if not re.match(pattern, val_str):
                if val_str not in unexpected_values:
                    unexpected_values[val_str] = []
                unexpected_values[val_str].append(idx)

        if unexpected_values:
            # Create summary DataFrame
            summary = []
            for value, indices in unexpected_values.items():
                length = len(value)

                # Diagnose issues
                issues = []
                if length < 3:
                    issues.append("too short")
                elif length > 4:
                    issues.append("too long")

                if length == 4 and value[0] not in ['-'] + list('0123456789'):
                    issues.append("invalid prefix")

                if length >= 3 and (not value[-3:].isupper() or not value[-3:].isalpha()):
                    issues.append("invalid format")

                # Get sample indices (first 5)
                sample_indices = indices[:5]

                summary.append({
                    'value': value,
                    'count': len(indices),
                    'length': length,
                    'issues': ', '.join(issues) if issues else 'invalid format',
                    'sample_indices': sample_indices
                })

            unexpected_df = pd.DataFrame(summary).sort_values('count', ascending=False)

            print(f"\n{'='*60}")
            print(f"VALIDATION: {column_name}")
            print(f"{'='*60}")
            print(f"Total rows with unexpected values: {sum(len(indices) for indices in unexpected_values.values())}")
            print(f"Unique unexpected values: {len(unexpected_values)}")
            print(f"\nUnexpected values (sorted by frequency):")
            print(unexpected_df.to_string(index=False))
            print(f"{'='*60}\n")

            return unexpected_df
        else:
            print(f"✓ Column '{column_name}': All values are in correct format")
            return pd.DataFrame()

    # 9.3.2 Clean building codes
    def building_codes(self, fill_value="unset"):
        """
        Clean the BUILDING CODES column by standardising various invalid values.
        Values containing 'TBD'/'TBC' or other 'later' indicators → 'LATER'
        Other invalid values → 'later'
        """
        column_name = 'BUILDING CODE'
        print(f"Column name: {column_name}")
        if column_name not in self.current_df.columns:
            print(f"⚠ Column '{column_name}' not found")
            return self.current_df

        # Patterns that mean "to be defined later"
        later_patterns = ['a définir', 'à définir', 'a definir', 'à definir', '?']
        unset_patterns = ['?']

        later_count = 0
        unset_count = 0

        for idx, val in self.current_df[column_name].items():
            if pd.isna(val):
                continue

            val_str = str(val).strip()
            val_lower = val_str.lower()

            # Check if any "later" pattern appears in the string
            if any(pattern in val_lower for pattern in later_patterns):
                self.current_df.at[idx, column_name] = 'LATER'
                later_count += 1
            if any(pattern in val_lower for pattern in unset_patterns):
                self.current_df.at[idx, column_name] = fill_value
                if fill_value == "LATER":
                    later_count += 1
                else:
                    unset_count += 1

        print(f"✓ Cleaned '{column_name}' column:")
        print(f"  {later_count} values → 'LATER'")
        if unset_count > 0:
            print(f"  {unset_count} values → 'unset'")

        return self.current_df
    
    # 9.4 Clean unexpected values    
    def find_null_like_values(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Find values that look like they should be null (e.g., '?', 'empty', 'unknown', etc.)

        Args:
            columns: List of column names to check (None = all object columns)

        Returns:
            DataFrame showing columns with null-like values
        """
        # Common null-like values
        null_like = [
            '?', '??', '???',
            'none', 'None', 'NONE',
            'null', 'Null', 'NULL',
            'nan', 'NaN', 'NAN',
            'empty', 'Empty', 'EMPTY', '[E]', '[e]'
            'unknown', 'Unknown', 'UNKNOWN',
            'missing', 'Missing', 'MISSING',
            'non trouvé', 'Non trouvé', 'NON TROUVÉ'
            '-', '--', '---',
            '', ' ', '  ',
        ]

        if columns is None:
            columns = self.current_df.select_dtypes(include=['object', 'string']).columns.tolist()

        results = []

        for col in columns:
            if col not in self.current_df.columns:
                continue
            
            # Create a mask for actual null values
            is_actually_null = self.current_df[col].isna()

            # Only convert non-null values into string
            series = self.current_df[col][~is_actually_null].astype(str)

            found_null_like = {}
            for null_val in null_like:
                count = (series.str.strip() == null_val).sum()
                if count > 0:
                    found_null_like[null_val] = count

            if found_null_like:
                for val, count in found_null_like.items():
                    pct = round(count / len(self.current_df) * 100, 2)
                    results.append({
                        'column': col,
                        'null_like_value': repr(val),
                        'count': count,
                        'percentage': pct
                    })

        if not results:
            print("✓ No null-like values found")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        print(f"\n{'='*100}")
        print("NULL-LIKE VALUES FOUND")
        print(f"{'='*100}")
        print(results_df.to_string(index=False))
        print(f"{'='*100}\n")

        return results_df
    
    def check_expected_values(self, column_name: str, expected_values: List, 
                             case_sensitive: bool = False) -> pd.DataFrame:
        """
        Check if a column contains only expected values.
        Returns a DataFrame of unexpected values.

        Args:
            column_name: Name of the column to check
            expected_values: List of expected values (NaN/None is automatically included)
            case_sensitive: Whether to do case-sensitive comparison for strings

        Returns:
            DataFrame with unexpected values, their counts, and sample row indices
        """
        if column_name not in self.current_df.columns:
            print(f"⚠ Column '{column_name}' not found")
            return pd.DataFrame()

        series = self.current_df[column_name]

        # Convert expected values for comparison
        if not case_sensitive and all(isinstance(v, str) or pd.isna(v) for v in expected_values):
            expected_set = {str(v).lower() if not pd.isna(v) else np.nan for v in expected_values}
            series_compare = series.apply(lambda x: str(x).lower() if not pd.isna(x) else np.nan)
        else:
            expected_set = set(expected_values)
            series_compare = series

        # Add NaN to expected values (always expected)
        expected_set.add(np.nan)

        # Find which expected values are present
        present_expected = []
        for exp_val in expected_values:
            if not case_sensitive and isinstance(exp_val, str):
                match_mask = series_compare == str(exp_val).lower()
            else:
                match_mask = series_compare == exp_val
            
            count = match_mask.sum()
            if count > 0:
                present_expected.append({
                    'expected_value': exp_val,
                    'count': count,
                    'percentage': round(count / len(self.current_df) * 100, 2)
                })
        
        # Count NaN values separately
        nan_count = series.isna().sum()
        if nan_count > 0:
            present_expected.append({
                'expected_value': 'NaN/None',
                'count': nan_count,
                'percentage': round(nan_count / len(self.current_df) * 100, 2)
            })
        
        # Find unexpected values
        unexpected_mask = ~series_compare.isin(expected_set) & series.notna()
        unexpected_values = series[unexpected_mask]

        if len(unexpected_values) == 0:
            print(f"\n{'='*100}")
            print(f"COLUMN: {column_name}")
            print(f"{'='*100}")
            print(f"✓ All values are as expected!\n")
            
            if present_expected:
                present_df = pd.DataFrame(present_expected)
                print("Expected values present:")
                print(present_df.to_string(index=False))
            
            print(f"{'='*100}\n")
            return pd.DataFrame()

        # Analyse unexpected values
        value_counts = unexpected_values.value_counts()

        results = []
        for value, count in value_counts.items():
            # Get sample indices
            indices = unexpected_values[unexpected_values == value].index.tolist()
            sample_indices = indices[:5]  # First 5 occurrences

            results.append({
                'unexpected_value': value,
                'count': count,
                'percentage': round(count / len(self.current_df) * 100, 2),
                'sample_row_indices': sample_indices
            })

        results_df = pd.DataFrame(results)

        # print(f"\n{'='*100}")
        # print(f"UNEXPECTED VALUES: {column_name}")
        # print(f"{'='*100}")
        # print(f"Expected values: {expected_values}")
        # print(f"Found {len(value_counts)} unexpected value(s) in {len(unexpected_values)} rows:\n")
        # print(results_df.to_string(index=False))
        # print(f"{'='*100}\n")

        print(f"\n{'='*100}")
        print(f"COLUMN: {column_name}")
        print(f"{'='*100}")
        
        # Display expected values present
        if present_expected:
            present_df = pd.DataFrame(present_expected)
            print("Expected values present:")
            print(present_df.to_string(index=False))
            print()
        
        # Display unexpected values
        print(f"Found {len(value_counts)} unexpected value(s) in {len(unexpected_values)} rows:")
        print(results_df.to_string(index=False))
        print(f"{'='*100}\n")
        
        return results_df

    def check_unexpected_multiple_columns(self, case_sensitive: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Check multiple columns at once using a dictionary of expected values.

        Args:
            case_sensitive: Whether to do case-sensitive comparison

        Returns:
            Dictionary mapping column names to their unexpected values DataFrames
        """
        results = {}
        expected_values_dict = self.EXPECTED_VALUES
        
        print(f"\n{'#'*100}")
        print(f"CHECKING {len(expected_values_dict)} COLUMNS FOR UNEXPECTED VALUES")
        print(f"{'#'*100}\n")

        for col_name, expected_vals in expected_values_dict.items():
            if col_name not in self.current_df.columns:
                print(f"⚠ Skipping '{col_name}' - column not found")
                continue
            
            unexpected_df = self.check_expected_values(col_name, expected_vals, case_sensitive)
            if len(unexpected_df) > 0:
                results[col_name] = unexpected_df

        # Summary
        print(f"\n{'#'*100}")
        print("SUMMARY")
        print(f"{'#'*100}")
        if results:
            print(f"Found unexpected values in {len(results)} column(s):")
            for col_name in results.keys():
                print(f"  - {col_name}")
        else:
            print("✓ All columns contain only expected values!")
        print(f"{'#'*100}\n")

        return results
    
    def convert_operated_season(self, val):
        """Convert various formats to 1/0/LATER"""
        if pd.isna(val):
            return val

        # Handle numeric types
        if isinstance(val, (int, float)):
            return "1" if val >= 0.5 else "0"

        # Handle strings
        val_str = str(val).strip().lower()

        if val_str in ['1', '1.0', 'yes', 'y', 'true']:
            return "1"
        elif val_str in ['0', '0.0', 'no', 'n', 'false']:
            return "0"
        elif val_str in ['later', 'pending']:
            return 'LATER'
        elif val_str in ['permanent', 'intermittent', 'rare', 'none', 'na']:
            return val_str.upper()
        else:
            # For decimal values with comma as separator or anything else unclear
            try:
                # Replace comma with dot for European decimal format
                numeric = float(val_str.replace(',', '.'))
                return "1" if numeric >= 0.5 else "0"
            except:
                return "0"
    
    def replace_values(self, column_name: str, replacement_map: Dict, 
                      case_sensitive: bool = False) -> pd.DataFrame:
        """
        Replace specific values in a column with new values.
        Can replace with NaN/None to set as null.

        Args:
            column_name: Name of the column to modify
            replacement_map: Dictionary mapping old values to new values
                            Use None or np.nan as the new value to set as null
            case_sensitive: Whether to do case-sensitive matching for strings

        Returns:
            The updated DataFrame
        """
        if column_name not in self.current_df.columns:
            print(f"⚠ Column '{column_name}' not found")
            return self.current_df

        # Count changes
        original_values = self.current_df[column_name].copy()

        # Special handling for OPERATED SUMMER/WINTER columns
        if not case_sensitive and column_name in ["OPERATED SUMMER", "OPERATED WINTER"]:
            self.current_df[column_name] = self.current_df[column_name].apply(self.convert_operated_season)
        elif case_sensitive:
            # Direct replacement
            self.current_df[column_name] = self.current_df[column_name].replace(replacement_map)
        else:
            # Case-insensitive replacement for string columns
            if self.current_df[column_name].dtype == 'object':
                # Create case-insensitive mapping
                def replace_case_insensitive(val):
                    if pd.isna(val):
                        return val
                    val_str = str(val)
                    for old, new in replacement_map.items():
                        if pd.isna(old):
                            continue
                        if val_str.lower() == str(old).lower():
                            return new
                    return val

                self.current_df[column_name] = self.current_df[column_name].apply(replace_case_insensitive)
            else:
                # For non-string columns, do direct replacement
                self.current_df[column_name] = self.current_df[column_name].replace(replacement_map)

        # Count changes
        changes = (original_values != self.current_df[column_name]).sum()

        # Handle comparison with NaN
        if changes == 0:
            both_nan = original_values.isna() & self.current_df[column_name].isna()
            actually_changed = (~both_nan) & (original_values != self.current_df[column_name])
            changes = actually_changed.sum()

        print(f"✓ Replaced values in '{column_name}': {changes} value(s) changed")

        # Show what was replaced
        for old_val, new_val in replacement_map.items():
            count = (original_values == old_val).sum() if not pd.isna(old_val) else original_values.isna().sum()
            if count > 0:
                new_display = 'NULL' if pd.isna(new_val) else repr(new_val)
                old_display = 'NULL' if pd.isna(old_val) else repr(old_val)
                print(f"  {old_display} → {new_display}: {count} occurrence(s)")

        return self.current_df

    def replace_multiple_columns(self, replacement_dict: Dict[str, Dict], 
                                 case_sensitive: bool = False) -> pd.DataFrame:
        """
        Replace values in multiple columns at once.

        Args:
            replacement_dict: Dictionary mapping column names to their replacement maps
                             Format: {'column_name': {'old_val': 'new_val', ...}, ...}
            case_sensitive: Whether to do case-sensitive matching

        Returns:
            The updated DataFrame
        """
        print(f"\n{'='*100}")
        print(f"REPLACING VALUES IN {len(replacement_dict)} COLUMN(S)")
        print(f"{'='*100}\n")

        for col_name, replace_map in replacement_dict.items():
            if col_name not in self.current_df.columns:
                print(f"⚠ Skipping '{col_name}' - column not found")
                continue
            
            self.replace_values(col_name, replace_map, case_sensitive)
            print()

        print(f"{'='*100}")
        print("✓ Replacement complete")
        print(f"{'='*100}\n")

        return self.current_df
    
    # 9.5 Swap NaN with `later`
    def get_missing_summary(self) -> pd.DataFrame:
        """
        Get a comprehensive summary of missing values across all columns.

        Returns:
            DataFrame with missing value statistics per column
        """
        missing_data = []

        for col in self.current_df.columns:
            missing_count = self.current_df[col].isna().sum()
            if missing_count > 0:
                missing_pct = round(missing_count / len(self.current_df) * 100, 2)
                dtype = str(self.current_df[col].dtype)

                missing_data.append({
                    'column': col,
                    'missing_count': missing_count,
                    'missing_percentage': missing_pct,
                    'dtype': dtype,
                    'total_rows': len(self.current_df)
                })

        if not missing_data:
            print("✓ No missing values in any column!")
            return pd.DataFrame()

        missing_df = pd.DataFrame(missing_data).sort_values('missing_count', ascending=False)

        print(f"\n{'='*100}")
        print("MISSING VALUES SUMMARY")
        print(f"{'='*100}")
        print(f"Columns with missing values: {len(missing_df)}/{len(self.current_df.columns)}")
        print(f"Total missing values: {missing_df['missing_count'].sum()}")
        print()
        print(missing_df.to_string(index=False))
        print(f"{'='*100}\n")

        return missing_df

    # 9.6 Clean textual columns (normally, only `-` is left as a weird value)
    def textual_columns(self, only_later):
        """
        later all variations of -, no, etc. string across all columns.
        """
        # List of N/A variations to replace
        values_to_clean = [
            '-', 'empty', 'à'
        ]

        # Get only categorical columns (object, string, category types)
        categorical_cols = self.current_df.select_dtypes(include=['object', 'string', 'category']).columns

        # Replace only in categorical columns
        for col in categorical_cols:
            for val in values_to_clean:
                # Case-insensitive replacement
                mask = self.current_df[col].astype(str).str.lower() == val.lower()
                if only_later:
                    self.current_df.loc[mask, col] = 'LATER'
                else:
                    self.current_df.loc[mask, col] = 'unset'
                    
        
        if only_later:
            self.current_df = self.current_df.replace("unset", "LATER")

        print(f"✓ Cleaned {len(categorical_cols)} categorical columns")
        return self.current_df
    
    
    # ---- Remove duplicates ---
    def remove_duplicates(self):
        """Removes duplicate rows from the current DataFrame."""
        initial_rows = len(self.current_df)
        self.current_df = self.current_df.drop_duplicates(ignore_index=True)
        removed_rows = initial_rows - len(self.current_df)
        print(f"✓ Removed {removed_rows} duplicate rows (kept {len(self.current_df)} unique rows)")
        return self.current_df
    
    
    # ---- Extract codes ---
    def extract_codes(self, only_ecs=False):
        """
        Extract prefix and suffix from code columns and insert them as new columns.

        Args:
            only_ecs: If True, only extract from CODE column. If False, extract from all code columns.

        Returns:
            The updated DataFrame
        """
        def extract_beginning_end(code):
            """Extract beginning and ending characters from an ID."""
            if pd.isna(code):
                return code, code

            code = str(code).strip()

            # Check if value is already a code or purely alphabetic
            if (len(code) <= 4 or
                (code.replace(" ", "").isalpha() and 
                 not any(code[i].lower() == 'x' for i in range(4, 8) if i < len(code)))):
                return code, "no suffix"
                
            # Check for weird lengths (comments/descriptions)
            if (12 < len(code) < 22) or len(code) > 25:
                return code, "no suffix"
            
            # Extract beginning based on first character
            if code[0].isdigit() or code[0] == '-':
                beginning = code[:6].upper()
            else:
                beginning = code[:5].upper()

            last_three = code[-3:].upper()

            # Check if last 3 are all numbers
            weird_chars = set('!@#$%^&*(){}[]<>?/\\|`~:;_"\',.')

            if last_three.isdigit() or any(char in weird_chars for char in last_three):
                # Check for format: [digit/-][letter][letter][letter][digit][digit][digit][digit][letter/digit/etc.]
                pattern = r'^[\d-][A-Za-z]{3}\d{4}[A-Za-z\d-]$'
                match = re.match(pattern, code)
            
                if match:
                    ending = match.group(1).upper()  # Everything after the first 8 characters
                else:
                    ending = "no suffix"
            elif last_three[0:1].isdigit():
                if last_three[1].isdigit() and last_three[2] == "-":
                    ending = "no suffix"
                else:
                    ending = last_three[1:]
            else:
                ending = last_three

            return beginning, ending

        to_drop = []
        
        # Process CODE column
        code_col = "ECS CODE"
        if code_col in self.current_df.columns:
            prefixes, suffixes = zip(*self.current_df[code_col].apply(extract_beginning_end))
            prefixes = pd.Series(prefixes, index=self.current_df.index)
            suffixes = pd.Series(suffixes, index=self.current_df.index)

            insert_pos = self.current_df.columns.get_loc(code_col) + 1
            self.current_df.insert(insert_pos, "ECS suffix", suffixes)
            self.current_df.insert(insert_pos, "ECS prefix", prefixes)
            
            to_drop.append(code_col)

        if not only_ecs:
            # Process additional columns
            columns_to_process = [
                ("CONSUMER ROOM", "CONSUMER ROOM prefix", "CONSUMER ROOM suffix"),
                ("POWER SOURCE ECS CODE", "Power source ECS prefix", "Power source ECS suffix"),
                ("PACKAGE ECS CODE", "PACKAGE ECS CODE prefix", "PACKAGE ECS CODE suffix"),
                ("INTERLOCKED LOADS", "INTERLOCKED LOADS prefix", "INTERLOCKED LOADS suffix"),
                ("SWITCHING DEVICE ECS CODE", "SWITCHING DEVICE ECS prefix", "SWITCHING DEVICE ECS suffix")
            ]

            for source_col, prefix_name, suffix_name in columns_to_process:
                if source_col in self.current_df.columns:
                    print(f"Creating prefixes for {source_col}...")
                    prefixes, suffixes = zip(*self.current_df[source_col].apply(extract_beginning_end))
                    if prefixes:
                        print("Extracted prefixes.")
                    if suffixes:
                        print("Extracted suffixes.")
                    else:
                        print("Extraction failed?")
                    prefixes = pd.Series(prefixes, index=self.current_df.index)
                    suffixes = pd.Series(suffixes, index=self.current_df.index)

                    insert_pos = self.current_df.columns.get_loc(source_col) + 1
                    self.current_df.insert(insert_pos, suffix_name, suffixes)
                    self.current_df.insert(insert_pos, prefix_name, prefixes)
                    
                    to_drop.append(source_col)
    
        if to_drop:
            self.current_df = self.current_df.drop(columns=to_drop)

        print(f"✓ Extracted codes from columns")
        return self.current_df

    ##########################################################################################################################################
    ##########################################################################################################################################
    ##########################################################################################################################################
    

    # ---- Handling Missing Values ----
    def analyse_column_values(self, column_name: str, top_n: int = 20) -> pd.DataFrame:
        """
        Analyse all unique values in a column with their counts and percentages.
        Useful for manually identifying unexpected values.

        Args:
            column_name: Name of the column to analyse
            top_n: Number of top values to show (None for all)

        Returns:
            DataFrame with value, count, and percentage
        """
        if column_name not in self.current_df.columns:
            print(f"⚠ Column '{column_name}' not found")
            return pd.DataFrame()

        # Get value counts including NaN
        value_counts = self.current_df[column_name].value_counts(dropna=False)

        # Calculate percentages
        total = len(self.current_df)
        percentages = (value_counts / total * 100).round(2)

        # Create analysis DataFrame
        analysis_df = pd.DataFrame({
            'value': value_counts.index,
            'count': value_counts.values,
            'percentage': percentages.values
        })

        # Show top N or all
        if top_n is not None:
            analysis_df = analysis_df.head(top_n)

        print(f"\n{'='*100}")
        print(f"VALUE ANALYSIS: {column_name}")
        print(f"{'='*100}")
        print(f"Total rows: {total}")
        print(f"Unique values: {self.current_df[column_name].nunique(dropna=False)}")
        print(f"Missing (NaN): {self.current_df[column_name].isna().sum()} ({(self.current_df[column_name].isna().sum()/total*100):.2f}%)")
        print(f"\nValue distribution:")
        print(analysis_df.to_string(index=False))
        print(f"{'='*100}\n")

        return analysis_df





    


    

    
    
    # ---- Summary and Utility Methods ----
    def get_summary(self, show_missing=False):
        """Get a summary of the current dataset state"""
        print("="*100)
        print("DATASET SUMMARY")
        print("="*100)
        print(f"Shape: {self.current_df.shape[0]} rows × {self.current_df.shape[1]} columns")
        print(f"\nData types:")
        print(self.current_df.dtypes.value_counts())
        if show_missing:
            print(f"\nMissing values:")
            missing = self.current_df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            if len(missing) > 0:
                for col, count in missing.items():
                    pct = (count / len(self.current_df)) * 100
                    print(f"  {col}: {count} ({pct:.1f}%)")
            else:
                print("  No missing values!")
    
    ##########################################################################################################################################
    ################################ Unused functions, but might come in handy for other projects ############################################
    ##########################################################################################################################################
    # ---- Type converstion methods ----
    def convert_to_boolean(self, columns: list, true_values: list = None, false_values: list = None):
        """
        Converts specified columns to boolean type.
        
        Args:
            columns: List of column names to convert
            true_values: List of values to consider as True (default: common true values)
            false_values: List of values to consider as False (default: common false values)
        """
        if true_values is None:
            true_values = [1, 1.0, '1', 'true', 't', 'yes', 'y', 'vrai', 'oui', 'o']
        if false_values is None:
            false_values = [0, 0.0, '0', 'false', 'f', 'no', 'n', 'faux', 'non']
        
        true_vals = {str(v).lower() for v in true_values}
        false_vals = {str(v).lower() for v in false_values}
        
        def map_to_bool(val):
            if pd.isna(val):
                return pd.NA
            s_val = str(val).lower().strip()
            if s_val in true_vals:
                return True
            if s_val in false_vals:
                return False
            return pd.NA
        
        converted = []
        for col in columns:
            if col not in self.current_df.columns:
                print(f"⚠ Column '{col}' not found, skipping")
                continue
            
            try:
                self.current_df[col] = self.current_df[col].apply(map_to_bool).astype('boolean')
                converted.append(col)
            except Exception as e:
                print(f"⚠ Error converting '{col}': {e}")
        
        print(f"✓ Converted {len(converted)} columns to boolean type")
        return self.current_df
    
    def convert_to_datetime(self, columns: list, format: str = None):
        """
        Converts specified columns to datetime type.
        
        Args:
            columns: List of column names to convert
            format: Optional datetime format string (e.g., '%Y-%m-%d')
        """
        converted = []
        for col in columns:
            if col not in self.current_df.columns:
                print(f"⚠ Column '{col}' not found, skipping")
                continue
            
            try:
                if format:
                    self.current_df[col] = pd.to_datetime(self.current_df[col], format=format, errors='coerce')
                else:
                    self.current_df[col] = pd.to_datetime(self.current_df[col], errors='coerce')
                converted.append(col)
            except Exception as e:
                print(f"⚠ Error converting '{col}': {e}")
        
        print(f"✓ Converted {len(converted)} columns to datetime type")
        return self.current_df
    
    
    # ---- XX ----
    