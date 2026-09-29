import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="2D FEM Structural Matrix Solver",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 2D Finite Element Matrix Solver & Truss/Frame Analyzer")
st.markdown("**Core Matrix Structural Analysis Engine** — Built from scratch in Python using NumPy and Matplotlib.")

st.sidebar.header("1. Structure Configuration")
analysis_type = st.sidebar.selectbox("Analysis Type", ["2D Truss (Axial Only)", "2D Beam / Frame (Flexural & Axial)"])

st.markdown("---")
st.subheader("Structure Modeling Parameters")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### **Nodes Definition**")
    num_nodes = st.number_input("Number of Nodes", min_value=2, max_value=10, value=3)
    
    nodes = {}
    for i in range(1, int(num_nodes) + 1):
        nc1, nc2 = st.columns(2)
        default_x = float((i-1)*5.0)
        default_y = 0.0 if i < 3 else 3.0
        x = nc1.number_input(f"Node {i} X (m)", value=default_x, key=f"nx_{i}")
        y = nc2.number_input(f"Node {i} Y (m)", value=default_y, key=f"ny_{i}")
        nodes[i] = (x, y)

with col2:
    node_keys = list(nodes.keys())
    st.markdown("### **Element Connectivity**")
    num_elements = st.number_input("Number of Elements", min_value=1, max_value=15, value=3)
    
    elements = {}
    for j in range(1, int(num_elements) + 1):
        ec1, ec2 = st.columns(2)
        # Give sensible default connections to avoid same-node errors
        default_end = node_keys[j % len(node_keys)]
        n1 = ec1.selectbox(f"Elem {j} Start Node", options=node_keys, key=f"e1_{j}")
        n2 = ec2.selectbox(f"Elem {j} End Node", options=node_keys, index=node_keys.index(default_end), key=f"e2_{j}")
        elements[j] = (n1, n2)

st.markdown("---")
st.subheader("Material & Support Conditions")

mc1, mc2 = st.columns(2)
with mc1:
    E = st.number_input("Young's Modulus, E (GPa)", value=200.0) * 1e9  # Pascals
    A = st.number_input("Cross-Sectional Area, A (m²)", value=0.01)
    
with mc2:
    st.markdown("**Support Fixity (1 = Fixed, 0 = Free)**")
    support_node = st.selectbox("Select Node for Support Pinning/Fixing", options=node_keys)
    fix_x = st.checkbox(f"Fix Node {support_node} in X", value=True)
    fix_y = st.checkbox(f"Fix Node {support_node} in Y", value=True)

st.markdown("---")
st.subheader("Applied Nodal Loads")
load_node = st.selectbox("Apply Load at Node", options=node_keys, key="load_node")
load_fx = st.number_input("Force in X Direction (kN)", value=0.0) * 1e3
load_fy = st.number_input("Force in Y Direction (kN)", value=-50.0) * 1e3

if st.button("Run FEM Matrix Analysis", type="primary"):
    
    # Validation check for zero-length elements
    zero_length_found = False
    for elem_id, (n1, n2) in elements.items():
        if n1 == n2:
            st.error(f"Error: Element {elem_id} connects Node {n1} to itself! Start and end nodes must be different.")
            zero_length_found = True
            break
            
    if not zero_length_found:
        dof_per_node = 2
        total_dof = int(num_nodes * dof_per_node)
        
        K = np.zeros((total_dof, total_dof))
        F = np.zeros(total_dof)
        
        # Apply Nodal Load Vector
        F[(load_node - 1) * 2] = load_fx
        F[(load_node - 1) * 2 + 1] = load_fy
        
        # Global Stiffness Matrix Assembly Loop (2D Truss elements)
        for elem_id, (n1, n2) in elements.items():
            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            
            L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            c = (x2 - x1) / L
            s = (y2 - y1) / L
            
            k_val = (E * A) / L
            
            k_global = k_val * np.array([
                [c*c, c*s, -c*c, -c*s],
                [c*s, s*s, -c*s, -s*s],
                [-c*c, -c*s, c*c, c*s],
                [-c*s, -s*s, c*s, s*s]
            ])
            
            dofs = [
                (n1 - 1) * 2,     
                (n1 - 1) * 2 + 1, 
                (n2 - 1) * 2,     
                (n2 - 1) * 2 + 1  
            ]
            
            for i in range(4):
                for j in range(4):
                    K[dofs[i], dofs[j]] += k_global[i, j]

        # Apply Boundary Conditions
        constrained_dofs = []
        if fix_x:
            constrained_dofs.append((support_node - 1) * 2)
        if fix_y:
            constrained_dofs.append((support_node - 1) * 2 + 1)
            
        # Ensure rigid body motion prevention (anchor Node 1 by default if unconstrained)
        if 0 not in constrained_dofs and 1 not in constrained_dofs and support_node != 1:
            constrained_dofs.extend([0, 1])

        free_dofs = [i for i in range(total_dof) if i not in constrained_dofs]
        
        # Partition stiffness matrix and force vector for free DOFs
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]
        
        try:
            d_f = np.linalg.solve(K_ff, F_f)
            
            displacements = np.zeros(total_dof)
            for idx, dof in enumerate(free_dofs):
                displacements[dof] = d_f[idx]
                
            st.success("Finite Element Matrix Solution Converged Successfully!")
            
            # Results display table
            st.subheader("Nodal Displacements Summary")
            disp_results = []
            for i in range(1, int(num_nodes) + 1):
                ux = displacements[(i-1)*2] * 1000  # convert meters to mm
                uy = displacements[(i-1)*2 + 1] * 1000
                disp_results.append({
                    "Node": i,
                    "Displacement X (mm)": round(ux, 4),
                    "Displacement Y (mm)": round(uy, 4)
                })
                
            st.dataframe(disp_results, use_container_width=True)
            
            # Professional Deformed vs. Undeformed Plot
            st.subheader("Deformed vs. Undeformed Structure Shape")
            fig, ax = plt.subplots(figsize=(9, 6))
            
            scale_factor = 500.0  # Amplification factor for visibility
            
            for n1, n2 in elements.values():
                x1, y1 = nodes[n1]
                x2, y2 = nodes[n2]
                
                # Original geometry (Black Dashed Line)
                ax.plot([x1, x2], [y1, y2], 'k--', lw=2, alpha=0.7, label="Undeformed Shape" if n1==node_keys[0] else "")
                
                # Deformed geometry (Red Solid Line with Scale Factor)
                dx1 = displacements[(n1 - 1) * 2] * scale_factor
                dy1 = displacements[(n1 - 1) * 2 + 1] * scale_factor
                dx2 = displacements[(n2 - 1) * 2] * scale_factor
                dy2 = displacements[(n2 - 1) * 2 + 1] * scale_factor
                
                ax.plot([x1 + dx1, x2 + dx2], [y1 + dy1, y2 + dy2], 'r-', lw=3, label="Deformed Shape (Scaled)" if n1==node_keys[0] else "")

            ax.set_title("Finite Element Deformed Mesh Analysis", fontsize=12, fontweight='bold', pad=12)
            ax.set_xlabel("X Coordinate (m)", fontsize=10, fontweight='bold')
            ax.set_ylabel("Y Coordinate (m)", fontsize=10, fontweight='bold')
            
            # Equal aspect ratio prevents geometric distortion
            ax.set_aspect('equal', adjustable='datalim')
            
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.legend(loc='upper right', frameon=True)
            st.pyplot(fig)

        except np.linalg.LinAlgError:
            st.error("Matrix is singular! Please verify that your support conditions are sufficient to prevent rigid body motion.")