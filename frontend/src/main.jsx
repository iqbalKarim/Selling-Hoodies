import React from "react"
import ReactDOM from "react-dom/client"
import App from "./App.jsx"
import "./index.css"
import { SelectedImagesContextProvider } from "./context/selectedImagesContext.jsx"
import ActiveModelContextProvider from "./context/activeModelContext.jsx"

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ActiveModelContextProvider>
      <SelectedImagesContextProvider>
        <App />
      </SelectedImagesContextProvider>
    </ActiveModelContextProvider>
  </React.StrictMode>
)
