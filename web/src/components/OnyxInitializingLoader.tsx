import { useContext } from "react";
import { SettingsContext } from "./settings/SettingsProvider";

export function OnyxInitializingLoader() {
  const settings = useContext(SettingsContext);

  return (
    <div className="mx-auto my-auto animate-pulse">
      <p className="text-lg text-text font-semibold">
        Initializing {settings?.enterpriseSettings?.application_name ?? "Onyx"}
      </p>
    </div>
  );
}
