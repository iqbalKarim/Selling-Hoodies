import React from "react"
import ReactDOM from "react-dom/client"
import App from "./App.jsx"
import "./index.css"
import { SelectedImagesContextProvider } from "./context/selectedImagesContext.jsx"

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <SelectedImagesContextProvider>
      <App />
    </SelectedImagesContextProvider>
  </React.StrictMode>
)
