import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "@/flygo-app";

import "@/index.css";

const rootElement = document.querySelector("#root");

if (!rootElement) {
  throw new Error("FlyGo root element was not found");
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>
);
