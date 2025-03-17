# Cluster Map Overview

This document explains how the `leonardo.tex` cluster map works. The key concepts are outlined below.

## Switch Identifier

The switch identifier (last column) is interpreted as follows:

- **Example:** `10102`  
  This identifier is decomposed into:
  - **(1):** Region
  - **(01):** Rack Local ID
  - **(02):** Switch Local ID

Thus, `10102` corresponds to **Region 1, first rack, second switch**.

## Switches

- The switches are the **leaf (L1)** switches.
- There are **3 switches per rack**, with local IDs that are even numbers: **00, 02, 04**.

## Region Concept

- The **region** is related to the power, cooling, and management systems.
- Typically, a region comprises **two Dragonfly Cells (or Groups)**.

## Additional Identifiers

- **Absolute Rack ID:**  
  An absolute rack identifier has been added, ranging from **1 to 138**.
- **Physical Row (ROW):**  
  Represents a physical row in the space.
- **Partition:**  
  A partition is used to distinguish between:
  - **CPU nodes:** (DataCentric/General Purpose partition)
  - **GPU nodes:** (Booster partition)
