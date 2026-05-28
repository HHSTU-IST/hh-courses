// 表格数据
const tableData = [
  { id: 3, name: "笔记本", quantity: 24, status: "在售" },
  { id: 1, name: "鼠标", quantity: 156, status: "在售" },
  { id: 2, name: "键盘", quantity: 89, status: "缺货" },
  { id: 4, name: "显示器", quantity: 12, status: "在售" },
  { id: 5, name: "耳机", quantity: 203, status: "在售" },
];

// 当前排序状态
let sortColumn = 0;
let sortAsc = true;
// 正在编辑的单元格信息
let editingCell = null;

// 初始化表格
function renderTable(data) {
  const tableBody = document.getElementById("tableBody");
  tableBody.innerHTML = "";

  data.forEach((item, rowIndex) => {
    const row = document.createElement("tr");
    // 定义各列对应的数据字段和类型
    const columns = [
      { field: "id", type: "number" },
      { field: "name", type: "string" },
      { field: "quantity", type: "number" },
      { field: "status", type: "string" },
    ];

    columns.forEach((col, colIndex) => {
      const cell = document.createElement("td");
      cell.textContent = item[col.field];
      cell.dataset.row = rowIndex;
      cell.dataset.col = colIndex;
      cell.dataset.field = col.field;
      cell.dataset.type = col.type;
      // 双击单元格进入编辑模式
      cell.ondblclick = startEditing;
      row.appendChild(cell);
    });

    tableBody.appendChild(row);
  });
}

// 开始编辑单元格
function startEditing(e) {
  const cell = e.target;
  // 若已经在编辑状态，不重复处理
  if (editingCell) return;

  editingCell = cell;
  const originalValue = cell.textContent;
  const input = document.createElement("input");
  input.type = cell.dataset.type === "number" ? "number" : "text";
  input.value = originalValue;
  input.className = "edit-input"; // 使用 CSS 类

  // 替换单元格内容为输入框
  cell.textContent = "";
  cell.appendChild(input);
  input.focus();

  // 失去焦点时保存
  input.onblur = () => saveEdit(cell, input, originalValue);
  // 按回车键保存
  input.onkeyup = (e) => {
    if (e.key === "Enter") {
      input.blur();
    }
  };
}

// 保存编辑内容
function saveEdit(cell, input, originalValue) {
  const newValue = input.value.trim();
  const rowIndex = parseInt(cell.dataset.row);
  const field = cell.dataset.field;
  const type = cell.dataset.type;

  // 数据验证
  if (newValue === "") {
    alert("内容不能为空");
    cell.textContent = originalValue;
    editingCell = null;
    return;
  }

  if (type === "number" && isNaN(Number(newValue))) {
    alert("请输入有效的数字");
    cell.textContent = originalValue;
    editingCell = null;
    return;
  }

  // 更新原始数据
  tableData[rowIndex][field] = type === "number" ? Number(newValue) : newValue;

  // 更新单元格显示
  cell.textContent = newValue;
  editingCell = null;
}

// 排序函数
function sortTable(column) {
  // 切换排序方向
  if (sortColumn === column) {
    sortAsc = !sortAsc;
  } else {
    sortColumn = column;
    sortAsc = true;
  }

  // 复制数据并排序
  const sortedData = [...tableData].sort((a, b) => {
    const values = [
      [a.id, b.id],
      [a.name, b.name],
      [a.quantity, b.quantity],
      [a.status, b.status],
    ];

    const valA = values[column][0];
    const valB = values[column][1];

    // 数字排序
    if (typeof valA === "number" && typeof valB === "number") {
      return sortAsc ? valA - valB : valB - valA;
    }

    // 字符串排序
    return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
  });

  renderTable(sortedData);
}

// 初始渲染
renderTable(tableData);
