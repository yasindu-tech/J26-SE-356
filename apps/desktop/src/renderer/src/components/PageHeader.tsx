import type { ReactNode } from "react";
import "./PageHeader.css";

/** The title row at the top of a screen: heading, one line of context, and its actions. */
export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="page-header">
      <div className="page-header__text">
        <h1 className="page-header__title">{title}</h1>
        {subtitle && <span className="page-header__subtitle">{subtitle}</span>}
      </div>
      {actions && <div className="page-header__actions">{actions}</div>}
    </div>
  );
}
