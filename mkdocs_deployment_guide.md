# Deploying a Static Site with MkDocs and GitHub Pages

## Prerequisites

- Python installed on your machine.
- A GitHub account with a repository for your project.

## Steps

1. **Install MkDocs**:

    ```bash
    pip install mkdocs
    pip install mkdocs-material
    pip install pymdown-extensions
    pip install mkdocs mkdocs-gitbook
    pip install mkdocs-glightbox
    ```

    If got error "pymdown-extensions", upgrade to the latest version:

    ```bash
       pip install --upgrade pymdown-extensions
    ```

2. **Initialize MkDocs Project**:

   ```bash
   mkdocs new my-project
   cd my-project
   ```

3. **Organize Markdown Files**:
   - Place all markdown files in the `docs` directory.

4. **Configure Navigation**:
   - Edit `mkdocs.yml` to define the site's navigation.

5. **Add number plugin**

    ```bash
    # Follow the guide here: https://github.com/ignorantshr/mkdocs-add-number-plugin
    pip install mkdocs-add-number-plugin
    ```

6. **Build and Serve Locally**:

   ```bash
   mkdocs build
   mkdocs serve
   ```

6. **Deploy to GitHub Pages**:

   ```bash
   mkdocs gh-deploy
   ```

## WindSurf Install Markdown Enhanced Preview
### Manually Download and Install Extensions
If you **cannot install extensions directly** in WindSurf due to restrictions, you can **manually download and install** them.

---

### **Step 1: Download the `.vsix` Extension File**

1. **Open the VS Code Marketplace**:
   [https://marketplace.visualstudio.com/vscode](https://marketplace.visualstudio.com/vscode)
2. **Search for the extension** you want to install.
3. Copy the extension’s URL and **modify it** to download the `.vsix` file:
   - Original Marketplace URL:

     ```
     https://marketplace.visualstudio.com/items?itemName=ms-python.python
     ```

   - Modify the URL:

     ```
     https://marketplace.visualstudio.com/_apis/public/gallery/publishers/ms-python/vsextensions/python/latest/vsix
     ```

4. **Open the modified URL in a browser** → The `.vsix` file will start downloading.

---

### **Step 2: Install the `.vsix` Extension in VS Code**
Once you have the `.vsix` file, install it in WindSurf:

1. Open **VS Code** in WindSurf.
2. Open the **Extensions Panel** (`Ctrl + Shift + X`).
3. Click on the **More Actions** `⋮` menu (top-right corner).
4. Select **"Install from VSIX..."**.
5. Choose the downloaded `.vsix` file and install.

✅ **Now the extension should be installed successfully!** 🚀

---

### **Alternative: Install via Command Line**
If the VS Code UI is restricted, try installing via the terminal:

```bash
code --install-extension /path/to/extension.vsix
```

Replace `/path/to/extension.vsix` with the actual path to the file.

## **📌 MkDocs Material Advanced Markdown Extensions**

### **1. Tabs**
Allows different content sections or code blocks to be displayed in tabs.

```markdown
=== "Python"
    ```python
    print("Hello, Python!")
    ```
=== "JavaScript"
    ```js
    console.log("Hello, JavaScript!");
    ```
```

✅ **Result**:
=== "Python"
    ```python
    print("Hello, Python!")
    ```
=== "JavaScript"
    ```js
    console.log("Hello, JavaScript!");
    ```

---

### **2. Admonitions (Callout Blocks)**
Provides colored blocks for **notes, warnings, tips, and success messages**.

```markdown
!!! note
    This is a general note.

!!! warning "Caution"
    This is a warning message.

!!! success "Completed"
    Task has been successfully completed!
```

✅ **Result**:
!!! note
    This is a general note.

!!! warning "Caution"
    This is a warning message.

!!! success "Completed"
    Task has been successfully completed!

---

### **3. Collapsible Sections**
Allows **content to be hidden and expanded when clicked**, useful for FAQs or additional explanations.

```markdown
??? note "Click to expand"
    This content is hidden by default.

???+ warning "Expanded by default"
    This section is visible by default but can be collapsed.
```

✅ **Result**:
??? note "Click to expand"
    This content is hidden by default.

???+ warning "Expanded by default"
    This section is visible by default but can be collapsed.

---

### **4. Task Lists**
Enables **checkable task lists** inside Markdown.

```markdown
- [x] Completed task
- [ ] Pending task
```

✅ **Result**:

- [x] Completed task
- [ ] Pending task

---

### **5. Code Block Enhancements**
#### **🌟 Show Line Numbers**

```yaml
markdown_extensions:
  - pymdownx.superfences
  - pymdownx.highlight:
      linenums: true
```

```markdown
```python
def add(a, b):
    return a + b

print(add(2, 3))
```

```

✅ **Result**:
```python
def add(a, b):
    return a + b

print(add(2, 3))
```

---

#### **🌟 Code Block with Titles**

```markdown
```python title="example.py"
print("Hello, World!")
```

```

✅ **Result**:
```python title="example.py"
print("Hello, World!")
```

---

#### **🌟 Highlight Specific Lines**

```markdown
```python hl_lines="2 4"
def add(a, b):
    return a + b  # This line is highlighted
print(add(2, 3))  # This line is highlighted
```

```

✅ **Result**:
```python hl_lines="2 4"
def add(a, b):
    return a + b  # This line is highlighted
print(add(2, 3))  # This line is highlighted
```

---

### **6. Text Formatting Enhancements**
#### **✅ Highlight Text**

```markdown
==Highlighted text==
```

✅ **Result**: ==Highlighted text==

---

#### **✅ Strikethrough**

```markdown
~~Strikethrough text~~
```

✅ **Result**: ~~Strikethrough text~~

---

### **7. Table of Contents (TOC)**

```markdown
[TOC]
```

✅ **Result**:
[TOC] will automatically generate a page table of contents (requires `toc.integrate` enabled).

---

### **8. Mathematical Equations (KaTeX / MathJax)**

```yaml
markdown_extensions:
  - pymdownx.arithmatex:
      generic: true
extra_javascript:
  - https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.4/katex.min.js
  - https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.4/contrib/auto-render.min.js
  - js/katex-init.js
```

```markdown

$$
\int_a^b f(x) dx
$$


```

✅ **Result**:
\[
\int_a^b f(x) dx
\]

---

### **9. Mermaid Diagrams (Flowcharts, Sequence Diagrams)**
Enable Mermaid.js for flowchart support:

```yaml
markdown_extensions:
  - pymdownx.superfences
extra_javascript:
  - https://cdnjs.cloudflare.com/ajax/libs/mermaid/9.3.0/mermaid.min.js
```

```markdown
```mermaid
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
```

```

✅ **Result**:
```mermaid
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
```

---

### **10. Variables and Snippets**
You can define **abbreviations, definitions, and formulas** in separate Markdown files and reference them.

### **📌 Organizing `includes/` directory**

```
includes/
 ├── abbreviations.md  # Abbreviations
 ├── definitions.md    # Definitions
 ├── formulas.md       # Mathematical Equations
```

#### **✅ Step 1: Define Variables**
📄 **`includes/abbreviations.md`**

```markdown
*[AI]: Artificial Intelligence
*[ML]: Machine Learning
*[NLP]: Natural Language Processing
```

📄 **`includes/definitions.md`**

```markdown
:deep_learning: Deep learning is a machine learning method based on artificial neural networks.
:gradient_descent: Gradient Descent is an optimization algorithm used to minimize loss functions.
```

📄 **`includes/formulas.md`**

```markdown
:einstein: $E=mc^2$
:bayes: $P(A|B) = \frac{P(B|A) P(A)}{P(B)}$

:integral: $$
\int_a^b f(x) dx
$$


```

---

#### **✅ Step 2: Configure `mkdocs.yml`**

```yaml
markdown_extensions:
  - pymdownx.snippets:
      auto_append:
        - includes/abbreviations.md
        - includes/definitions.md
        - includes/formulas.md
  - pymdownx.arithmatex  # Math support
```

---

#### **✅ Step 3: Use Variables in Markdown**

```markdown
Artificial Intelligence (:AI:) and :deep_learning: are closely related.
Einstein's energy equation is :einstein:, and Bayes' theorem is :bayes:.
```

✅ **Result**:
> Artificial Intelligence (**Artificial Intelligence**) and **Deep learning** are closely related.
> Einstein's energy equation is $E=mc^2$, and Bayes' theorem is $P(A|B) = \frac{P(B|A) P(A)}{P(B)}$.

---

### **11. Icons Support**
You can insert Material Design icons in your documentation.

```markdown
:material-home:
:material-email:
```

✅ **Result**:
:material-home:
:material-email:

---

### **12. Buttons**
You can create interactive buttons inside your documentation.

```markdown
[Click Me](#){ .md-button }
[Primary](#){ .md-button .md-button--primary }
[Secondary](#){ .md-button .md-button--secondary }
```

✅ **Result**:
[Click Me](#){ .md-button }
[Primary](#){ .md-button .md-button--primary }
[Secondary](#){ .md-button .md-button--secondary }

---

### **🌟 Final Summary**

| Feature | Syntax |
|---------|--------|
| ✅ Tabs | `=== "Python" ...` |
| ✅ Admonitions | `!!! warning` |
| ✅ Collapsible Sections | `??? note` |
| ✅ Task Lists | `- [x] Task` |
| ✅ Code Line Numbers | `linenums: true` |
| ✅ Highlight Text | `==highlight==` |

| ✅ Math Equations | `$$ E=mc^2 $$` |


| ✅ Mermaid Diagrams | ` ```mermaid` |
| ✅ Variable References | `:einstein:` |

## Additional Tips

- Ensure your GitHub repository is set up to use GitHub Pages.
- Check the `gh-pages` branch for the deployed site content.
